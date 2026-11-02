//! Observe backend initialization, rather than mistaking an available GPU for a used GPU.
use std::cell::RefCell;
use std::ffi::CStr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use whisper_rs::whisper_rs_sys as sys;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CpuReason {
    GpuDisabled,
    NoBackendCompiled,
    NoDeviceFound,
    InitFailed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WhisperDevice {
    Gpu { backend: String, name: String },
    Cpu(CpuReason),
    Unknown,
}

impl WhisperDevice {
    pub(crate) fn json(&self) -> serde_json::Value {
        match self {
            Self::Unknown => serde_json::json!({"kind": "unknown"}),
            Self::Gpu { backend, name } => {
                serde_json::json!({"kind": "gpu", "backend": backend, "name": name})
            }
            Self::Cpu(reason) => serde_json::json!({"kind": "cpu", "reason": match reason {
                CpuReason::GpuDisabled => "gpu_disabled",
                CpuReason::NoBackendCompiled => "no_backend_compiled",
                CpuReason::NoDeviceFound => "no_device_found",
                CpuReason::InitFailed => "init_failed",
            }, "description": self.to_string()}),
        }
    }
}

impl std::fmt::Display for WhisperDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown => write!(f, "device unknown (GPU use could not be confirmed)"),
            Self::Gpu { backend, name } => write!(f, "GPU ({backend}, {name})"),
            Self::Cpu(reason) => write!(
                f,
                "CPU ({})",
                match reason {
                    CpuReason::GpuDisabled => "GPU disabled",
                    CpuReason::NoBackendCompiled => "no GPU backend compiled",
                    CpuReason::NoDeviceFound => "no GPU found",
                    CpuReason::InitFailed => "GPU initialization failed",
                }
            ),
        }
    }
}

#[derive(Default)]
struct Evidence {
    selected: Option<String>,
    failed: bool,
    no_device: bool,
}
thread_local! { static CAPTURE: RefCell<Option<Evidence>> = const { RefCell::new(None) }; }
static LAST: Mutex<Option<WhisperDevice>> = Mutex::new(None);
static DAEMON: AtomicBool = AtomicBool::new(false);
static WARNED: AtomicBool = AtomicBool::new(false);

pub(crate) fn daemon_started() {
    DAEMON.store(true, Ordering::Relaxed);
}
pub(crate) fn last() -> Option<WhisperDevice> {
    LAST.lock().expect("Whisper device lock poisoned").clone()
}
pub(crate) fn begin() {
    CAPTURE.with(|capture| *capture.borrow_mut() = Some(Evidence::default()));
}

fn observe(evidence: &mut Evidence, line: &str) {
    if let Some(name) = line
        .strip_prefix("whisper_backend_init_gpu: using ")
        .and_then(|s| s.trim().strip_suffix(" backend"))
    {
        evidence.selected = Some(name.to_owned());
    }
    if line.contains("whisper_backend_init_gpu: failed to initialize ") {
        evidence.failed = true;
    }
    if line.contains("whisper_backend_init_gpu: no GPU found") {
        evidence.no_device = true;
    }
}

pub(crate) fn log_line(line: &str) {
    CAPTURE.with(|capture| {
        if let Some(evidence) = capture.borrow_mut().as_mut() {
            for line in line.lines() {
                observe(evidence, line);
            }
        }
    });
}

fn classify(
    enabled: bool,
    compiled: bool,
    evidence: &Evidence,
    selected: Option<WhisperDevice>,
) -> WhisperDevice {
    if !enabled {
        WhisperDevice::Cpu(CpuReason::GpuDisabled)
    } else if !compiled {
        WhisperDevice::Cpu(CpuReason::NoBackendCompiled)
    } else if evidence.failed {
        WhisperDevice::Cpu(CpuReason::InitFailed)
    } else if let Some(device) = selected {
        device
    } else if evidence.no_device {
        WhisperDevice::Cpu(CpuReason::NoDeviceFound)
    } else {
        WhisperDevice::Unknown
    }
}

// Called only after decoder-state initialization has completed. The registry owns all
// strings and devices for the process lifetime; we copy strings before returning.
fn selected_device(name: &str) -> Option<WhisperDevice> {
    unsafe {
        for i in 0..sys::ggml_backend_dev_count() {
            let dev = sys::ggml_backend_dev_get(i);
            let kind = sys::ggml_backend_dev_type(dev);
            if kind != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                && kind != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
            {
                continue;
            }
            if CStr::from_ptr(sys::ggml_backend_dev_name(dev)).to_string_lossy() == name {
                return Some(WhisperDevice::Gpu {
                    backend: CStr::from_ptr(sys::ggml_backend_reg_name(
                        sys::ggml_backend_dev_backend_reg(dev),
                    ))
                    .to_string_lossy()
                    .into_owned(),
                    name: CStr::from_ptr(sys::ggml_backend_dev_description(dev))
                        .to_string_lossy()
                        .into_owned(),
                });
            }
        }
    }
    None
}

/// Whisper's `gpu_device` index for the preferred GPU. whisper.cpp numbers
/// only GPU and integrated-GPU devices, in registry order, and uses index 0 by
/// default, which on hybrid laptops is often the slower integrated GPU. Prefer
/// the first discrete GPU; otherwise keep whisper.cpp's default.
fn preferred_gpu_index(kinds: &[sys::ggml_backend_dev_type]) -> i32 {
    kinds
        .iter()
        .filter(|kind| {
            **kind == sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                || **kind == sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
        })
        .position(|kind| *kind == sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU)
        .map_or(0, |index| index as i32)
}

/// The `gpu_device` to request from whisper.cpp on this machine.
pub(crate) fn preferred_gpu_device() -> i32 {
    // SAFETY: the ggml registry is process-global and lives for the process;
    // these calls only enumerate it.
    let kinds: Vec<_> = unsafe {
        (0..sys::ggml_backend_dev_count())
            .map(|i| sys::ggml_backend_dev_type(sys::ggml_backend_dev_get(i)))
            .collect()
    };
    preferred_gpu_index(&kinds)
}

fn warn_once(warned: &AtomicBool, fallback: bool) -> bool {
    fallback && !warned.swap(true, Ordering::Relaxed)
}

pub(crate) fn finish(enabled: bool, report_load: bool) {
    let evidence =
        CAPTURE.with(|capture| capture.borrow_mut().take().expect("device capture started"));
    let selected = evidence.selected.as_deref().and_then(selected_device);
    let compiled = cfg!(any(
        target_os = "windows",
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ));
    let device = classify(enabled, compiled, &evidence, selected);
    if !report_load && last().as_ref() == Some(&device) {
        return;
    }
    let line = format!("Whisper: {device}");
    eprintln!("{line}");
    let daemon = DAEMON.load(Ordering::Relaxed);
    if daemon {
        crate::daemon::mirror_to_log(&line);
    }
    let fallback = enabled && !matches!(device, WhisperDevice::Gpu { .. });
    if fallback {
        let target = if matches!(device, WhisperDevice::Unknown) {
            "unconfirmed device"
        } else {
            "CPU"
        };
        let warning =
            format!("warning: fallback: true; attempted=GPU; fallback={target}; reason={device}");
        eprintln!("{warning}");
        if daemon {
            crate::daemon::mirror_to_log(&warning);
        }
    }
    if daemon && warn_once(&WARNED, fallback) {
        crate::notify::show_forced(
            crate::notify::Kind::Info,
            "Whisper GPU unavailable",
            &format!("{line}. Settings > General shows the last Whisper device."),
        );
    }
    *LAST.lock().expect("Whisper device lock poisoned") = Some(device);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initialization_evidence_and_classification() {
        let mut evidence = Evidence::default();
        observe(
            &mut evidence,
            "whisper_backend_init_gpu: using Vulkan0 backend\n",
        );
        assert_eq!(evidence.selected.as_deref(), Some("Vulkan0"));
        let gpu = WhisperDevice::Gpu {
            backend: "Vulkan".into(),
            name: "Test GPU".into(),
        };
        assert_eq!(classify(true, true, &evidence, Some(gpu.clone())), gpu);
        assert_eq!(gpu.to_string(), "GPU (Vulkan, Test GPU)");
        assert_eq!(
            gpu.json(),
            serde_json::json!({"kind":"gpu", "backend":"Vulkan", "name":"Test GPU"})
        );
        observe(
            &mut evidence,
            "whisper_backend_init_gpu: failed to initialize Vulkan0 backend\n",
        );
        assert_eq!(
            classify(true, true, &evidence, Some(gpu)),
            WhisperDevice::Cpu(CpuReason::InitFailed)
        );
        assert_eq!(
            classify(false, true, &evidence, None).to_string(),
            "CPU (GPU disabled)"
        );
        assert_eq!(
            classify(true, false, &evidence, None),
            WhisperDevice::Cpu(CpuReason::NoBackendCompiled)
        );
        assert_eq!(
            classify(true, true, &Evidence::default(), None),
            WhisperDevice::Unknown
        );
        observe(&mut evidence, "whisper_backend_init_gpu: no GPU found\n");
        evidence.failed = false;
        assert_eq!(
            classify(true, true, &evidence, None).to_string(),
            "CPU (no GPU found)"
        );
    }
    #[test]
    fn prefers_the_first_discrete_gpu() {
        use sys::{
            ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_ACCEL as ACCEL,
            ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU as CPU,
            ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU as GPU,
            ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU as IGPU,
        };
        // Hybrid laptop: integrated GPU first, discrete second.
        assert_eq!(preferred_gpu_index(&[IGPU, GPU, CPU]), 1);
        // CPU/accelerator devices do not count toward whisper.cpp's index.
        assert_eq!(preferred_gpu_index(&[CPU, ACCEL, IGPU, CPU, GPU]), 1);
        assert_eq!(preferred_gpu_index(&[GPU, IGPU]), 0);
        assert_eq!(preferred_gpu_index(&[IGPU]), 0);
        assert_eq!(preferred_gpu_index(&[CPU]), 0);
        assert_eq!(preferred_gpu_index(&[]), 0);
    }

    #[test]
    fn notification_once_only_for_fallback() {
        let warned = AtomicBool::new(false);
        assert!(!warn_once(&warned, false));
        assert!(warn_once(&warned, true));
        assert!(!warn_once(&warned, true));
    }
}
