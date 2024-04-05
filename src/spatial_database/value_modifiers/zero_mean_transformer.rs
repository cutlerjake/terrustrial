pub struct ZeroMeanTransformer {
    mean: f32,
}

impl ZeroMeanTransformer {
    fn forward_transformer<T>(&self) -> impl Fn(&mut [f32]) {
        let mean = self.mean;
        move |data: &mut [f32]| {
            data.iter_mut().for_each(|d| *d -= mean);
        }
    }

    fn backward_transformer(&self) -> impl Fn(&mut [f32]) {
        let mean = self.mean;
        move |data: &mut [f32]| {
            data.iter_mut().for_each(|d| *d += mean);
        }
    }
}
