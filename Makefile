.PHONY: build test test-all install shards clean

build:
	cargo build --release

test:
	cargo test --lib

# Integration tests spawn ./target/debug/ouro-agent, so build the binary first.
test-all:
	cargo build --bin ouro-agent
	cargo test --workspace

install: build
	cargo install --path .

shards:
	python3 tools/shard_model.py $(MODEL) $(NODES) --output-dir shards

clean:
	cargo clean
