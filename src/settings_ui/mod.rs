//! Settings window: an egui page inside the daemon's winit event loop.
//!
//! Created on demand (tray `Settings…` or `dictate settings`), destroyed
//! completely on close (window, GL context and egui state are dropped), so
//! there is zero cost while closed. A second open request focuses the existing
//! window. Frames run on `RedrawRequested` and whenever egui asks; the loop
//! keeps `ControlFlow::Wait` (plus a `WaitUntil` deadline while egui has a
//! delayed repaint pending), so an idle daemon still sleeps.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use egui::ViewportId;
use glutin::context::{ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, WindowSurface};
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::raw_window_handle::HasWindowHandle;
use winit::window::{Window, WindowAttributes, WindowId};

pub(crate) mod page;

/// Window title.
pub(crate) const SETTINGS_TITLE: &str = "Dictate Settings";
/// Fixed inner size in logical pixels (min = max, not resizable), so tiling
/// window managers float the dialog instead of tiling it.
pub(crate) const FIXED_SIZE: (f64, f64) = (480.0, 640.0);
/// Wayland app id / X11 class, so users can write window rules.
/// Linux-only, like the `with_name` call that uses it.
#[cfg(target_os = "linux")]
pub(crate) const APP_ID: &str = "dictate";

/// Shortcut-recording state for the Settings page: whether the Record button
/// is capturing, the last capture error shown inline, and whether modifiers
/// were held with no key yet (modifier-only detection). Lives with the window,
/// so closing the window resets it.
#[derive(Debug, Default)]
pub(crate) struct ShortcutCapture {
    pub recording: bool,
    pub error: Option<String>,
    pub mods_seen: bool,
}

/// The open Settings window: winit window, glutin GL context and surface, and
/// egui state. Everything lives on the event-loop thread and is dropped on
/// close. [`Drop`] frees the GL objects while the context is still current.
pub(crate) struct SettingsWindow {
    window: Window,
    gl_context: PossiblyCurrentContext,
    /// Held so the GL display stays alive with the context. Never read,
    /// hence the underscore.
    _gl_display: glutin::display::Display,
    gl_surface: Surface<WindowSurface>,
    /// Held so the glow context stays alive with the painter's clone. Never
    /// read, hence the underscore.
    _gl: Arc<egui_glow::glow::Context>,
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    painter: egui_glow::Painter,
    viewport_info: egui::ViewportInfo,
    pub(crate) capture: ShortcutCapture,
}

impl SettingsWindow {
    /// Create the window, GL context and egui state on the event-loop thread.
    /// The window follows the system light/dark theme (egui's default, fed by
    /// `egui_winit` from the winit theme).
    pub(crate) fn open(event_loop: &ActiveEventLoop) -> anyhow::Result<Self> {
        let attributes = WindowAttributes::default()
            .with_title(SETTINGS_TITLE)
            .with_inner_size(LogicalSize::new(FIXED_SIZE.0, FIXED_SIZE.1))
            .with_min_inner_size(LogicalSize::new(FIXED_SIZE.0, FIXED_SIZE.1))
            .with_max_inner_size(LogicalSize::new(FIXED_SIZE.0, FIXED_SIZE.1))
            .with_resizable(false)
            .with_transparent(false);
        // One call sets the shared app-id field for both backends (the
        // Wayland and X11 `with_name` impls write the same field).
        #[cfg(target_os = "linux")]
        let attributes = winit::platform::wayland::WindowAttributesExtWayland::with_name(
            attributes, APP_ID, APP_ID,
        );
        let template = glutin::config::ConfigTemplateBuilder::new()
            .with_depth_size(0)
            .with_stencil_size(0)
            .with_alpha_size(0);
        // The picker has no error channel by design (glutin's own examples
        // expect here too): a working GL display always enumerates configs,
        // so an empty pick means a broken driver past successful GL init.
        let (window, gl_config) = glutin_winit::DisplayBuilder::new()
            .with_preference(glutin_winit::ApiPreference::FallbackEgl)
            .with_window_attributes(Some(attributes.clone()))
            .build(event_loop, template, |mut configs| {
                configs
                    .next()
                    .expect("no OpenGL configs offered by the driver")
            })
            .map_err(|err| {
                anyhow::anyhow!("Could not create the Settings window graphics display: {err}")
            })?;
        let gl_display = gl_config.display();
        let raw_window_handle = window
            .as_ref()
            .map(|window| {
                window
                    .window_handle()
                    .map(|handle| handle.as_raw())
                    .map_err(|err| {
                        anyhow::anyhow!("Could not use the Settings window handle: {err}")
                    })
            })
            .transpose()?;
        let context_attributes = ContextAttributesBuilder::new().build(raw_window_handle);
        let fallback_attributes = ContextAttributesBuilder::new()
            .with_context_api(glutin::context::ContextApi::Gles(None))
            .build(raw_window_handle);
        // SAFETY: the display, config and window handle are alive, and the
        // context is only made current on this thread below.
        let not_current = unsafe { gl_display.create_context(&gl_config, &context_attributes) }
            .or_else(|first| {
                // SAFETY: same as above; a GLES context when desktop GL fails.
                unsafe { gl_display.create_context(&gl_config, &fallback_attributes) }.map_err(
                    |second| {
                        anyhow::anyhow!(
                            "Could not create an OpenGL context for the Settings window: {first}; GLES fallback: {second}"
                        )
                    },
                )
            })?;
        // The window exists already (the attributes were passed to the
        // builder); finalize it only if a backend skipped that step.
        let window = match window {
            Some(window) => window,
            None => glutin_winit::finalize_window(event_loop, attributes, &gl_config)
                .map_err(|err| anyhow::anyhow!("Could not create the Settings window: {err}"))?,
        };
        let (width, height): (u32, u32) = window.inner_size().into();
        let surface_attributes = glutin::surface::SurfaceAttributesBuilder::<WindowSurface>::new()
            .build(
                window
                    .window_handle()
                    .map_err(|err| {
                        anyhow::anyhow!("Could not use the Settings window handle: {err}")
                    })?
                    .as_raw(),
                NonZeroU32::new(width).unwrap_or(NonZeroU32::MIN),
                NonZeroU32::new(height).unwrap_or(NonZeroU32::MIN),
            );
        // SAFETY: the display, config and window handle are alive.
        let gl_surface =
            unsafe { gl_display.create_window_surface(&gl_config, &surface_attributes) }.map_err(
                |err| anyhow::anyhow!("Could not create the Settings window surface: {err}"),
            )?;
        let gl_context = not_current.make_current(&gl_surface).map_err(|err| {
            anyhow::anyhow!("Could not activate the Settings window graphics context: {err}")
        })?;
        if let Err(err) = gl_surface.set_swap_interval(
            &gl_context,
            glutin::surface::SwapInterval::Wait(NonZeroU32::MIN),
        ) {
            // Vsync is a nicety; the window works without it.
            crate::daemon::log_line(format_args!(
                "warning: could not enable vsync for the Settings window: {err}"
            ));
        }
        let gl = {
            let gl_display = gl_display.clone();
            // SAFETY: the loader only resolves GL symbols through the live
            // display; a name with an interior NUL (impossible for GL symbols)
            // resolves to null, which glow treats as unavailable.
            unsafe {
                egui_glow::glow::Context::from_loader_function(|name| {
                    std::ffi::CString::new(name)
                        .map(|cname| gl_display.get_proc_address(&cname))
                        .unwrap_or(std::ptr::null())
                })
            }
        };
        let gl = Arc::new(gl);
        let painter = egui_glow::Painter::new(Arc::clone(&gl), "", None, true)
            .context("Could not initialize the Settings window renderer")?;
        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            event_loop,
            None,
            event_loop.system_theme(),
            Some(painter.max_texture_side()),
        );
        Ok(Self {
            window,
            gl_context,
            _gl_display: gl_display,
            gl_surface,
            _gl: gl,
            egui_ctx,
            egui_winit,
            painter,
            viewport_info: egui::ViewportInfo::default(),
            capture: ShortcutCapture::default(),
        })
    }

    /// This window's id, for routing `window_event`s.
    pub(crate) fn window_id(&self) -> WindowId {
        self.window.id()
    }

    /// Bring the window to the front (a second open request, and the initial
    /// open on macOS, where the app is Accessory and would stay behind).
    pub(crate) fn focus(&self) {
        self.window.focus_window();
    }

    /// Schedule a repaint.
    pub(crate) fn request_redraw(&self) {
        self.window.request_redraw();
    }

    /// Resize the GL surface (a zero side means minimized: skip it).
    pub(crate) fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        let width = NonZeroU32::new(size.width).unwrap_or(NonZeroU32::MIN);
        let height = NonZeroU32::new(size.height).unwrap_or(NonZeroU32::MIN);
        self.gl_surface.resize(&self.gl_context, width, height);
    }

    /// Feed one window event to egui. True when egui wants a repaint.
    pub(crate) fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        self.egui_winit.on_window_event(&self.window, event).repaint
    }

    /// Turn egui's clipboard shortcuts (Cmd+C/X/V, Shift+Delete, ...) on or
    /// off: off while a shortcut is recording, so capture sees plain
    /// `Event::Key`. Called after every frame, since egui-winit maps the keys
    /// when it receives them, not when input is taken.
    pub(crate) fn set_clipboard_shortcuts(&mut self, enabled: bool) {
        self.egui_winit.set_clipboard_shortcuts(enabled);
    }

    /// Run one frame: egui layout, platform output, paint, present. Returns
    /// the repaint delay egui asked for (`Duration::ZERO` = another frame
    /// immediately, `Duration::MAX` = nothing pending).
    pub(crate) fn redraw(
        &mut self,
        event_loop: &ActiveEventLoop,
        run_ui: impl FnMut(&mut egui::Ui),
    ) -> anyhow::Result<Duration> {
        let raw_input = self.egui_winit.take_egui_input(&self.window);
        let egui::FullOutput {
            mut platform_output,
            mut textures_delta,
            shapes,
            pixels_per_point,
            viewport_output,
        } = self.egui_ctx.run_ui(raw_input, run_ui);
        // Open clicked links through the daemon's opener (with error
        // reporting) instead of egui-winit's silent one.
        let mut urls = Vec::new();
        platform_output.commands.retain(|command| {
            if let egui::OutputCommand::OpenUrl(open) = command {
                urls.push(open.url.clone());
                false
            } else {
                true
            }
        });
        let mut repaint_delay = Duration::MAX;
        for (_, output) in viewport_output {
            repaint_delay = repaint_delay.min(output.repaint_delay);
            let mut actions = Vec::new();
            egui_winit::process_viewport_commands(
                &self.egui_ctx,
                &mut self.viewport_info,
                output.commands,
                &self.window,
                &mut actions,
            );
            for action in actions {
                crate::daemon::log_line(format_args!(
                    "warning: unsupported Settings window action: {action:?}"
                ));
            }
        }
        self.egui_winit.handle_platform_output_with_event_loop(
            &self.window,
            event_loop,
            platform_output,
        );
        for url in urls {
            if let Err(err) = crate::tray::open_target(&url) {
                let message = format!("{err:#}");
                crate::daemon::log_line(format_args!("error: {message}"));
                crate::notify::show(crate::notify::Kind::Error, "dictate", &message);
            }
        }
        let clipped = self.egui_ctx.tessellate(shapes, pixels_per_point);
        let (width, height): (u32, u32) = self.window.inner_size().into();
        let clear = self.egui_ctx.global_style().visuals.panel_fill;
        self.painter
            .clear([width, height], clear.to_normalized_gamma_f32());
        self.painter.paint_and_update_textures(
            [width, height],
            pixels_per_point,
            &clipped,
            &mut textures_delta,
        );
        self.gl_surface
            .swap_buffers(&self.gl_context)
            .context("Could not present the Settings window")?;
        Ok(repaint_delay)
    }
}

impl Drop for SettingsWindow {
    fn drop(&mut self) {
        // Free the GL objects while the context is still current; the context
        // itself is destroyed with the remaining fields right after.
        self.painter.destroy();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_window_title_and_size() {
        assert_eq!(SETTINGS_TITLE, "Dictate Settings");
        assert_eq!(FIXED_SIZE, (480.0, 640.0));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn settings_window_app_id() {
        assert_eq!(APP_ID, "dictate");
    }
}
