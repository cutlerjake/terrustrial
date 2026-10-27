//! Optional native portable GPU runtime. Failure to obtain a device is explicit;
//! callers can keep using the CPU engine. No browser event-loop support is implied.
use crate::{execution::Cancellation, Error, Result};
use std::{
    sync::{mpsc, Arc, Mutex},
    time::{Duration, Instant},
};

/// Shared device/queue. The adapter identity is part of a reproducibility record.
pub struct Context {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    failure: Arc<Mutex<Option<String>>>,
}
impl Context {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|error| Error::BackendUnavailable(error.to_string()))?;
        let limits = wgpu::Limits {
            max_buffer_size: adapter.limits().max_buffer_size,
            max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("terrustrial compute"),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|error| Error::BackendUnavailable(error.to_string()))?;
        let failure = Arc::new(Mutex::new(None));
        let captured = failure.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            *captured
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(error.to_string());
        }));
        Ok(Self {
            device,
            queue,
            info: adapter.get_info(),
            failure,
        })
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }
    pub(crate) fn check(&self) -> Result<()> {
        if let Some(error) = &*self
            .failure
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            return Err(Error::Backend(error.clone()));
        }
        Ok(())
    }
    /// Capture validation, allocation and internal backend errors as Results.
    pub(crate) fn checked<T>(&self, operation: impl FnOnce() -> T) -> Result<T> {
        self.check()?;
        let allocation = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = operation();
        let errors = [
            pollster::block_on(validation.pop()),
            pollster::block_on(internal.pop()),
            pollster::block_on(allocation.pop()),
        ];
        if let Some(error) = errors.into_iter().flatten().next() {
            return Err(Error::Backend(error.to_string()));
        }
        self.check()?;
        Ok(result)
    }
    pub(crate) fn readback(&self, buffer: &wgpu::Buffer, cancel: &Cancellation) -> Result<Vec<u8>> {
        let (sender, receiver) = mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let start = Instant::now();
        let result = loop {
            if let Err(error) = cancel.check().and_then(|_| self.check()) {
                break Err(error);
            }
            match receiver.try_recv() {
                Ok(result) => break result.map_err(|error| Error::Backend(error.to_string())),
                Err(mpsc::TryRecvError::Disconnected) => {
                    break Err(Error::Backend("GPU map callback disconnected".into()))
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
            if start.elapsed() > Duration::from_secs(30) {
                break Err(Error::Backend(
                    "GPU readback timed out after 30 seconds".into(),
                ));
            }
            match self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_millis(20)),
            }) {
                Ok(_) | Err(wgpu::PollError::Timeout) => {}
                Err(error) => break Err(Error::Backend(error.to_string())),
            }
        };
        if let Err(error) = result {
            buffer.unmap();
            return Err(error);
        }
        let bytes = buffer
            .slice(..)
            .get_mapped_range()
            .map(|view| view.to_vec())
            .map_err(|error| Error::Backend(error.to_string()));
        buffer.unmap();
        bytes
    }
}
