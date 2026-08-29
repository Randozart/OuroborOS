use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest_dir.parent().unwrap().to_path_buf();
    let bitnet = workspace.join("bitnet-cpp");
    let llama_root = bitnet.join("3rdparty/llama.cpp");
    let lib_dir = bitnet.join("build").join("bin");

    let clang_include = format!(
        "-I{}",
        llama_root.join("include").display()
    );
    let ggml_include = format!(
        "-I{}",
        llama_root.join("ggml").join("include").display()
    );

    let bindings = bindgen::Builder::default()
        .header(llama_root.join("include/llama.h").display().to_string())
        .clang_arg(&clang_include)
        .clang_arg(&ggml_include)
        .clang_arg("-std=c11")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        .layout_tests(false)
        .generate()
        .expect("bindgen failed to generate llama.h bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    let build_root = bitnet.join("build");
    let libllama_a = build_root.join("3rdparty/llama.cpp/src/libllama.a");

    if libllama_a.exists() {
        // Static link (preferred for worker nodes)
        println!("cargo:rustc-link-search=native={}", build_root.join("3rdparty/llama.cpp/src").display());
        println!("cargo:rustc-link-search=native={}", build_root.join("3rdparty/llama.cpp/ggml/src").display());
        println!("cargo:rustc-link-lib=static=llama");
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
        println!("cargo:rustc-link-lib=static=ggml-base");
    } else {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=llama");
        println!("cargo:rustc-link-lib=dylib=ggml");
        println!("cargo:rustc-link-lib=dylib=ggml-cpu");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    println!("cargo:rerun-if-changed={}", llama_root.join("include/llama.h").display());
}
