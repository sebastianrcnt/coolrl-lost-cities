use std::path::PathBuf;

fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    std::env::set_var("PROTOC", protoc);

    let proto_dir = PathBuf::from("../schemas");
    let proto_file = proto_dir.join("lost_cities.proto");

    println!("cargo:rerun-if-changed={}", proto_file.display());

    tonic_build::configure()
        .compile_protos(&[proto_file], &[proto_dir])
        .expect("compile lost_cities proto");
}
