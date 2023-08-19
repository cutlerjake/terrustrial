use crate::spatial_database::gridded_databases::GriddedDataBaseInterface;

pub struct Template {
    offsets: Vec<[isize; 3]>,
}

impl Template {
    pub fn new(offsets: Vec<[isize; 3]>) -> Self {
        Self { offsets }
    }

    #[inline(always)]
    pub fn get_ind(ind: &[usize; 3], offset: [isize; 3], bounds: [usize; 3]) -> Option<[usize; 3]> {
        let ind = [
            (ind[0] as isize + offset[0]),
            (ind[1] as isize + offset[1]),
            (ind[2] as isize + offset[2]),
        ];

        if ind[0] < 0 || ind[1] < 0 || ind[2] < 0 {
            return None;
        }

        if ind[0] >= bounds[0] as isize
            || ind[1] >= bounds[1] as isize
            || ind[2] >= bounds[2] as isize
        {
            return None;
        }
        Some(ind.map(|v| v as usize))
    }

    pub fn replicate_at_ind<GDB, T>(&self, db: &GDB, ind: &[usize; 3]) -> Option<Vec<T>>
    where
        GDB: GriddedDataBaseInterface<T>,
    {
        let mut replicates = vec![db.data_at_ind(ind)?];
        let shape = db.shape();
        for offset in self.offsets.iter() {
            if let Some(ind) = Self::get_ind(ind, *offset, shape) {
                if let Some(data) = db.data_at_ind(&ind) {
                    replicates.push(data);
                }
            } else {
                return None;
            }
        }

        Some(replicates)
    }
}
