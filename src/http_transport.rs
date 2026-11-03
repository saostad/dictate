//! One request per connection, with bounded intake and socket deadlines.
use std::io::{self, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};
use tiny_http::{HTTPVersion, Header, Method, Response};

pub(crate) struct Server(TcpListener);
impl Server {
    pub(crate) fn http(address: &str) -> io::Result<Self> {
        let listener = TcpListener::bind(address)?;
        listener.set_nonblocking(true)?;
        Ok(Self(listener))
    }
    pub(crate) fn server_addr(&self) -> SocketAddr {
        self.0.local_addr().expect("bound listener address")
    }
    pub(crate) fn recv_timeout(&self, poll: Duration) -> io::Result<Option<Request>> {
        match self.0.accept() {
            // macOS and Windows hand out accepted sockets that inherit the
            // listener's non-blocking mode; deadlines need blocking reads.
            Ok((stream, _)) => {
                if let Err(e) = stream.set_nonblocking(false) {
                    // Drop only this connection; the listener stays up.
                    eprintln!("error: HTTP connection setup failed: {e}");
                    return Ok(None);
                }
                Ok(Some(Request {
                    input: Body {
                        reader: BufReader::new(stream),
                        remaining: 0,
                        deadline: None,
                    },
                    method: Method::Get,
                    path: String::new(),
                    headers: Vec::new(),
                    headers_complete: false,
                    linger: None,
                }))
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                std::thread::park_timeout(poll);
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }
}
struct Body {
    reader: BufReader<TcpStream>,
    remaining: usize,
    deadline: Option<Instant>,
}
impl Body {
    fn read_socket(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if let Some(deadline) = self.deadline {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .filter(|d| !d.is_zero())
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::TimedOut, "request read deadline exceeded")
                })?;
            self.reader.get_ref().set_read_timeout(Some(remaining))?;
        }
        self.reader.read(buffer)
    }
}
impl Read for Body {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let len = buffer.len().min(self.remaining);
        if len == 0 {
            return Ok(0);
        }
        let count = self.read_socket(&mut buffer[..len])?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete body",
            ));
        }
        self.remaining -= count;
        Ok(count)
    }
}
pub(crate) struct Request {
    input: Body,
    method: Method,
    path: String,
    headers: Vec<Header>,
    headers_complete: bool,
    linger: Option<(usize, Duration)>,
}
impl Request {
    pub(crate) fn configure_close(&mut self, bytes: usize, timeout: Duration) {
        self.linger = Some((bytes, timeout));
    }

    pub(crate) fn read_headers(&mut self, timeout: Duration, limit: usize) -> io::Result<()> {
        self.input.deadline = Some(Instant::now().checked_add(timeout).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "read timeout is too large")
        })?);
        self.input
            .reader
            .get_ref()
            .set_write_timeout(Some(timeout))?;
        let mut bytes = Vec::new();
        while !bytes.ends_with(b"\r\n\r\n") {
            if bytes.len() >= limit {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "headers exceed limit",
                ));
            }
            let mut byte = [0];
            if self.input.read_socket(&mut byte)? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "incomplete headers",
                ));
            }
            bytes.push(byte[0]);
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid headers"))?;
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "invalid request headers");
        let mut lines = text.split("\r\n");
        let mut first = lines.next().ok_or_else(invalid)?.split_whitespace();
        self.method = first
            .next()
            .ok_or_else(invalid)?
            .parse()
            .map_err(|_| invalid())?;
        self.path = first.next().ok_or_else(invalid)?.to_owned();
        if !matches!(first.next(), Some("HTTP/1.0" | "HTTP/1.1")) || first.next().is_some() {
            return Err(invalid());
        }
        let mut length = None;
        for line in lines.take_while(|line| !line.is_empty()) {
            let header: Header = line.parse().map_err(|_| invalid())?;
            if header.field.equiv("Transfer-Encoding") {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "use Content-Length, not Transfer-Encoding",
                ));
            }
            if header.field.equiv("Content-Length") {
                if length.is_some() {
                    return Err(invalid());
                }
                length = Some(
                    header
                        .value
                        .as_str()
                        .parse::<usize>()
                        .map_err(|_| invalid())?,
                );
            }
            if header.field.equiv("Expect") {
                if !header.value.as_str().eq_ignore_ascii_case("100-continue") {
                    return Err(invalid());
                }
                self.input
                    .reader
                    .get_mut()
                    .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
            }
            self.headers.push(header);
        }
        if self.method == Method::Post && length.is_none() {
            return Err(invalid());
        }
        self.headers_complete = true;
        self.input.remaining = length.unwrap_or(0);
        Ok(())
    }
    pub(crate) fn url(&self) -> &str {
        &self.path
    }
    pub(crate) fn method(&self) -> &Method {
        &self.method
    }
    pub(crate) fn headers(&self) -> &[Header] {
        &self.headers
    }
    pub(crate) fn body_length(&self) -> Option<usize> {
        Some(self.input.remaining)
    }
    pub(crate) fn as_reader(&mut self) -> &mut dyn Read {
        &mut self.input
    }
    pub(crate) fn respond<R: Read>(mut self, response: Response<R>) -> io::Result<()> {
        let (drain_bytes, timeout) = self.linger.expect("configured connection close");
        // Overflow requests have not read headers, so also need a write deadline.
        if self.input.deadline.is_none() {
            self.input
                .reader
                .get_ref()
                .set_write_timeout(Some(timeout))?;
        }
        let needs_drain = !self.headers_complete || self.input.remaining != 0;
        let response =
            response.with_header(Header::from_bytes("Connection", "close").expect("fixed header"));
        if needs_drain {
            // Send a complete rejection in one write. Fragmented header writes
            // can remain in Nagle's buffer and be lost when an unread upload
            // forces a reset as the connection closes.
            let mut bytes = Vec::new();
            response.raw_print(
                &mut bytes,
                HTTPVersion(1, 1),
                &self.headers,
                self.method == Method::Head,
                None,
            )?;
            self.input.reader.get_mut().write_all(&bytes)?;
        } else {
            response.raw_print(
                self.input.reader.get_mut(),
                HTTPVersion(1, 1),
                &self.headers,
                self.method == Method::Head,
                None,
            )?;
        }
        // Send FIN only after the entire response. Keep receiving so an upload
        // racing the rejection cannot turn the response into a TCP reset.
        self.input.reader.get_ref().shutdown(Shutdown::Write)?;
        if needs_drain {
            self.input.deadline = Some(Instant::now().checked_add(timeout).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "linger timeout is too large")
            })?);
            let mut left = drain_bytes;
            let mut buffer = [0; 8192];
            while left > 0 {
                let len = left.min(buffer.len());
                match self.input.read_socket(&mut buffer[..len]) {
                    Ok(0) => break,
                    Ok(n) => left -= n,
                    Err(e)
                        if matches!(
                            e.kind(),
                            io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                        ) =>
                    {
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rejected_upload(
        drain_bytes: usize,
        timeout: Duration,
    ) -> (TcpStream, std::thread::JoinHandle<io::Result<()>>) {
        let server = Server::http("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(server.server_addr()).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let handler = std::thread::spawn(move || {
            let mut request = loop {
                if let Some(request) = server.recv_timeout(Duration::from_millis(1)).unwrap() {
                    break request;
                }
            };
            request.configure_close(drain_bytes, timeout);
            request.read_headers(Duration::from_secs(3), 16384)?;
            request.respond(
                Response::from_string(r#"{"error":{"kind":"too_large"}}"#).with_status_code(413),
            )
        });
        (client, handler)
    }
    #[test]
    fn incomplete_upload_receives_full_json_and_fin() {
        let (mut client, handler) = rejected_upload(1048576, Duration::from_secs(2));
        let mut request = b"POST / HTTP/1.1\r\nContent-Length: 100000\r\n\r\n".to_vec();
        // Exceed BufReader's read-ahead so unread bytes also remain in the socket.
        request.extend_from_slice(&[b'x'; 32768]);
        client.write_all(&request).unwrap();
        // Keep the upload half open: the response and FIN must arrive first.
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 413"), "{response}");
        let (_, body) = response.split_once("\r\n\r\n").unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(body).unwrap()["error"]["kind"],
            "too_large"
        );
        client.shutdown(Shutdown::Write).unwrap();
        handler.join().unwrap().unwrap();
    }
    #[test]
    fn continuous_upload_cannot_extend_absolute_linger_deadline() {
        let (mut client, handler) = rejected_upload(usize::MAX, Duration::from_millis(50));
        client
            .write_all(b"POST / HTTP/1.1\r\nContent-Length: 1000000000\r\n\r\nx")
            .unwrap();
        let mut upload = client.try_clone().unwrap();
        upload
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let writer = std::thread::spawn(move || while upload.write_all(&[0; 4096]).is_ok() {});
        let (done, completion) = std::sync::mpsc::channel();
        let joiner = std::thread::spawn(move || done.send(handler.join().unwrap()).unwrap());
        completion
            .recv_timeout(Duration::from_secs(2))
            .expect("upload extended linger deadline")
            .unwrap();
        drop(client);
        writer.join().unwrap();
        joiner.join().unwrap();
    }
    #[test]
    fn expired_body_deadline_fails_without_waiting_for_data() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (stream, _) = listener.accept().unwrap();
        let mut body = Body {
            reader: BufReader::new(stream),
            remaining: 1,
            deadline: Some(Instant::now()),
        };
        assert_eq!(
            body.read(&mut [0]).unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
    }
}
