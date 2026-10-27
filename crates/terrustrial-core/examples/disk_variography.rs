//! Local streaming sample-file construction and exact variogram measurements.
#[allow(dead_code)]
#[path = "support/training.rs"]
mod training;
use std::{fs::File, hint::black_box, path::Path, time::Instant};
use terrustrial_core::{
    data::Point3,
    execution::{Cancellation, MemoryBudget},
    grid::Shape,
    storage::grid::ingest::Rows as GridRows,
    variography::{
        disk::{self, BuildOptions, SampleFile},
        indexed::Workspace,
        Bin,
    },
    Error, Result,
};

fn fingerprint(bins: &[Bin]) -> Vec<(u64, Option<u64>, Option<u64>)> {
    bins.iter()
        .map(|b| {
            (
                b.pairs,
                b.mean_distance.map(f64::to_bits),
                b.semivariance.map(f64::to_bits),
            )
        })
        .collect()
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let cancel = Cancellation::default();
    let budget = MemoryBudget::new(128 << 20);
    match args.get(1).map(String::as_str) {
        Some("build") if args.len()==4 => {
            let start=Instant::now();
            let options=BuildOptions { variables:3,cache_bytes:16<<20,maximum_database_bytes:3<<30,budget:&budget,cancellation:&cancel };
            let report=if let Some(n)=args[2].strip_prefix("synthetic:") {
                let n: u64=n.parse()?;
                if n>6000000 { return Err("synthetic fixture capped at six million rows".into()); }
                let mut i=0;
                let mut source=|values:&mut[Option<f64>]| -> Result<_> {
                    if i==n { return Ok(None); }
                    let point=Point3::new([i as f64,(i%7) as f64*0.03,(i%11) as f64*0.02])?;
                    values.copy_from_slice(&[Some((i%17) as f64-8.),Some(4.-(i%17) as f64*0.5),Some(i as f64*0.25)]);
                    i+=1;
                    Ok(Some((u64::MAX-i+1,point)))
                };
                disk::build(&mut source,Path::new(&args[3]),options)?
            } else {
                let shape=Shape::new([150,200,200])?;
                let mut input=training::ReservoirRows::new(File::open(&args[2])?,shape,&["porosity.dat","acoustic_impedance.dat","density.dat"])?;
                let mut count=0;
                let mut source=|values:&mut[Option<f64>]| -> Result<_> {
                    let Some(id)=input.next(values)? else {
                        if count!=shape.cells() { return Err(Error::InvalidInput("incomplete reservoir grid".into())); }
                        return Ok(None);
                    };
                    let point=Point3::new([(id%150) as f64,((id/150)%200) as f64,(id/30000) as f64])?;
                    count+=1;
                    if count%500000==0 { eprintln!("imported {count} reservoir rows"); }
                    Ok(Some((id,point)))
                };
                disk::build(&mut source,Path::new(&args[3]),options)?
            };
            println!("metric,value\nsamples,{}\nvariables,{}\nunindexed_samples,{}\ndatabase_bytes,{}\nbuild_seconds,{}",report.samples,report.variables,report.unindexed_samples,report.database_bytes,start.elapsed().as_secs_f64());
        }
        Some("query") if args.len()==8 => {
            let readers:usize=args[3].parse()?;
            let iterations:usize=args[4].parse()?;
            let radius:f64=args[5].parse()?;
            let variables=[args[6].parse()?,args[7].parse()?];
            if !(1..=8).contains(&readers) || !(1..=1000000).contains(&iterations) || !radius.is_finite() || radius*0.5 <= 0. { return Err("invalid readers, iterations or two-bin radius".into()); }
            let pool=rayon::ThreadPoolBuilder::new().num_threads(readers).build()?;
            let start=Instant::now();
            let mut file=SampleFile::open(Path::new(&args[2]),readers,(32<<20)/readers as u64,&budget,&cancel)?;
            let mut workspace=Workspace::new(2,32,&budget)?;
            let open_seconds=start.elapsed().as_secs_f64();
            eprintln!("validated {} samples with {readers} readers",file.len());
            pool.install(|| -> Result<()> {
                let edges=[radius*0.5,radius];
                let first=file.experimental(variables,&edges,None,&mut workspace,&cancel)?;
                let expected=fingerprint(first.bins);
                let bins=first.bins.to_vec();
                let candidates=first.candidate_pairs;
                eprintln!("first exact evaluation complete");
                let start=Instant::now();
                for _ in 0..iterations {
                    let result=file.experimental(variables,&edges,None,&mut workspace,&cancel)?;
                    for (bin,expected) in result.bins.iter().zip(&expected) {
                        assert_eq!((bin.pairs,bin.mean_distance.map(f64::to_bits),bin.semivariance.map(f64::to_bits)),*expected);
                    }
                    black_box(result);
                }
                let seconds=start.elapsed().as_secs_f64();
                println!("metric,value\nsamples,{}\nreaders,{readers}\niterations,{iterations}\nradius,{radius}\nvariable_a,{}\nvariable_b,{}\ncandidate_pairs,{candidates}\nreserved_bytes,{}\nopen_seconds,{open_seconds}\nquery_seconds,{seconds}",file.len(),variables[0],variables[1],budget.used());
                for (i,bin) in bins.iter().enumerate() {
                    println!("bin_{i}_pairs,{}\nbin_{i}_distance,{}\nbin_{i}_gamma,{}",bin.pairs,bin.mean_distance.unwrap_or(0.),bin.semivariance.unwrap_or(0.));
                }
                Ok(())
            })?;
        }
        _ => return Err("usage: disk_variography build INPUT.csv|synthetic:N NEW_DB | query DB READERS ITERATIONS RADIUS VAR_A VAR_B".into()),
    }
    Ok(())
}
