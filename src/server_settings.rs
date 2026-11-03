//! Persisted HTTP service limits. CLI flags override these for one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServerSettings {
    pub listen: String,
    pub token_file: String,
    pub max_concurrency: usize,
    pub max_queue: usize,
    pub max_handlers: usize,
    pub read_timeout_secs: u64,
    pub max_header_bytes: usize,
    pub max_upload_bytes: usize,
    pub max_body_bytes: usize,
    pub max_duration_secs: u64,
    pub idle_unload_secs: u64,
    pub accept_poll_ms: u64,
    pub retry_after_secs: u64,
}
impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:7342".into(),
            token_file: String::new(),
            max_concurrency: 1,
            max_queue: 4,
            max_handlers: 16,
            read_timeout_secs: 60,
            max_header_bytes: 16384,
            max_upload_bytes: 52428800,
            max_body_bytes: 52559872,
            max_duration_secs: 600,
            idle_unload_secs: 1200,
            accept_poll_ms: 10,
            retry_after_secs: 1,
        }
    }
}
impl ServerSettings {
    pub(crate) const KEYS: &[&str] = &[
        "server.listen",
        "server.token_file",
        "server.max_concurrency",
        "server.max_queue",
        "server.max_handlers",
        "server.read_timeout_secs",
        "server.max_header_bytes",
        "server.max_upload_bytes",
        "server.max_body_bytes",
        "server.max_duration_secs",
        "server.idle_unload_secs",
        "server.accept_poll_ms",
        "server.retry_after_secs",
    ];
    pub(crate) fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        let mut next = self.clone();
        match key.trim_start_matches("server.") {
            "listen" => next.listen = value.to_owned(),
            "token_file" => next.token_file = value.to_owned(),
            "max_concurrency" => {
                next.max_concurrency = value
                    .parse()
                    .map_err(|_| "Expected a nonnegative integer")?
            }
            "max_queue" => {
                next.max_queue = value
                    .parse()
                    .map_err(|_| "Expected a nonnegative integer")?
            }
            "max_handlers" => {
                next.max_handlers = value
                    .parse()
                    .map_err(|_| "Expected a nonnegative integer")?
            }
            "read_timeout_secs" => {
                next.read_timeout_secs = value
                    .parse()
                    .map_err(|_| "Expected a nonnegative integer")?
            }
            "max_header_bytes" => {
                next.max_header_bytes = value
                    .parse()
                    .map_err(|_| "Expected a nonnegative integer")?
            }
            "max_upload_bytes" => {
                next.max_upload_bytes = value
                    .parse()
                    .map_err(|_| "Expected a nonnegative integer")?
            }
            "max_body_bytes" => {
                next.max_body_bytes = value
                    .parse()
                    .map_err(|_| "Expected a nonnegative integer")?
            }
            "max_duration_secs" => {
                next.max_duration_secs = value
                    .parse()
                    .map_err(|_| "Expected a nonnegative integer")?
            }
            "idle_unload_secs" => {
                next.idle_unload_secs = value
                    .parse()
                    .map_err(|_| "Expected a nonnegative integer")?
            }
            "accept_poll_ms" => {
                next.accept_poll_ms = value
                    .parse()
                    .map_err(|_| "Expected a nonnegative integer")?
            }
            "retry_after_secs" => {
                next.retry_after_secs = value
                    .parse()
                    .map_err(|_| "Expected a nonnegative integer")?
            }
            _ => return Err("Unknown server setting".into()),
        }
        next.validate()?;
        *self = next;
        Ok(())
    }
    pub(crate) fn validate(&self) -> Result<(), String> {
        self.listen
            .parse::<std::net::SocketAddr>()
            .map_err(|_| "listen must be ADDR:PORT")?;
        if self.token_file.contains(['\n', '\r', '\0']) {
            return Err("Invalid token file path".into());
        }
        if self.max_concurrency == 0 {
            return Err("max_concurrency must be positive".into());
        }
        if self.max_handlers == 0 {
            return Err("max_handlers must be positive".into());
        }
        if self.read_timeout_secs == 0 {
            return Err("read_timeout_secs must be positive".into());
        }
        if self.max_header_bytes == 0 {
            return Err("max_header_bytes must be positive".into());
        }
        if self.max_upload_bytes == 0 {
            return Err("max_upload_bytes must be positive".into());
        }
        if self.max_body_bytes == 0 {
            return Err("max_body_bytes must be positive".into());
        }
        if self.max_duration_secs == 0 {
            return Err("max_duration_secs must be positive".into());
        }
        if self.accept_poll_ms == 0 {
            return Err("accept_poll_ms must be positive".into());
        }
        if self.retry_after_secs == 0 {
            return Err("retry_after_secs must be positive".into());
        }
        for seconds in [self.read_timeout_secs, self.idle_unload_secs] {
            if std::time::Instant::now()
                .checked_add(std::time::Duration::from_secs(seconds))
                .is_none()
            {
                return Err("Server timeout exceeds the platform clock range".into());
            }
        }
        if self.max_concurrency > 256 || self.max_handlers > 4096 || self.max_queue > 65536 {
            return Err("Server worker, handler or queue limit is too large".into());
        }
        // Contract consumers use IEEE-754 JSON numbers for these limits.
        let max_exact_json_integer = (1_u64 << 53) - 1;
        if self.max_duration_secs > max_exact_json_integer
            || self.max_upload_bytes as u64 > max_exact_json_integer
        {
            return Err("Public limits must be exactly representable JSON integers".into());
        }
        if self.max_body_bytes > isize::MAX as usize || self.max_upload_bytes > isize::MAX as usize
        {
            return Err("Byte limits exceed addressable memory".into());
        }
        Ok(())
    }
    pub(crate) fn value(&self, key: &str) -> Result<String, String> {
        Ok(match key.trim_start_matches("server.") {
            "listen" => self.listen.to_string(),
            "token_file" => self.token_file.to_string(),
            "max_concurrency" => self.max_concurrency.to_string(),
            "max_queue" => self.max_queue.to_string(),
            "max_handlers" => self.max_handlers.to_string(),
            "read_timeout_secs" => self.read_timeout_secs.to_string(),
            "max_header_bytes" => self.max_header_bytes.to_string(),
            "max_upload_bytes" => self.max_upload_bytes.to_string(),
            "max_body_bytes" => self.max_body_bytes.to_string(),
            "max_duration_secs" => self.max_duration_secs.to_string(),
            "idle_unload_secs" => self.idle_unload_secs.to_string(),
            "accept_poll_ms" => self.accept_poll_ms.to_string(),
            "retry_after_secs" => self.retry_after_secs.to_string(),
            _ => return Err("Unknown server setting".into()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_and_round_trips_every_server_setting() {
        let mut settings = crate::config::Settings::default();
        for key in ServerSettings::KEYS {
            let value = settings.server.value(key).unwrap();
            settings.server.set(key, &value).unwrap();
        }
        for (key, value) in [
            ("max_concurrency", "0"),
            ("read_timeout_secs", "0"),
            ("listen", "localhost"),
            ("max_upload_bytes", "-1"),
            ("max_duration_secs", "NaN"),
            ("unknown", "1"),
        ] {
            assert!(settings.server.set(key, value).is_err());
        }
        settings.server.set("max_queue", "0").unwrap();
        settings.server.set("idle_unload_secs", "0").unwrap();
        settings.server.set("token_file", "a \\\"token").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        crate::config::save_to(&path, &settings).unwrap();
        assert_eq!(
            crate::config::parse(&std::fs::read_to_string(path).unwrap()).unwrap(),
            settings
        );
    }
}
