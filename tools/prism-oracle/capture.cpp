#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

// OuroborOS oracle capture: decode a prompt with the Prism fork, dump the
// full logits vector of the last prompt token plus any graph tensors whose
// name matches a prefix list. Rust side (ouro_cluster) replays the same
// token through Qwen35Model and compares (CONTRACTS.md L3).

struct CapState {
    bool active = false;              // capture only during the target decode
    std::vector<std::string> prefixes;
    std::string cap_path;
    std::ofstream out;
};

static bool want(const CapState & st, const char * name) {
    for (auto & p : st.prefixes) {
        if (strncmp(name, p.c_str(), p.size()) == 0) {
            return true;
        }
    }
    return false;
}

static void write_rec(std::ofstream & o, const char * name, const float * data, size_t n) {
    uint32_t nl = strlen(name);
    o.write((const char *) &nl, 4);
    o.write(name, nl);
    o.write((const char *) &n, 8);
    o.write((const char *) data, n * sizeof(float));
}

static bool cap_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * st = (CapState *) ud;
    if (ask) {
        return st->active && t->type == GGML_TYPE_F32 && want(*st, t->name);
    }
    if (st->active && want(*st, t->name)) {
        write_rec(st->out, t->name, (const float *) t->data, ggml_nelements(t));
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt = "Hello";
    std::string out_path = "oracle_logits.f32";
    std::string dump = "";
    int ngl = 0;
    int n_threads = 8;
    int n_predict = 1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-m") model_path = argv[++i];
        else if (a == "-p") prompt = argv[++i];
        else if (a == "-o") out_path = argv[++i];
        else if (a == "-d") dump = argv[++i];
        else if (a == "-ngl") ngl = std::stoi(argv[++i]);
        else if (a == "-t") n_threads = std::stoi(argv[++i]);
        else if (a == "-n") n_predict = std::stoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> toks(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);

    printf("tokens:");
    for (auto t : toks) printf(" %d", t);
    printf("\n");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_prompt + n_predict + 8;
    cp.n_batch = (uint32_t) n_prompt;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf = true;

    CapState st;
    if (!dump.empty()) {
        std::stringstream ss(dump);
        std::string p;
        while (std::getline(ss, p, ',')) st.prefixes.push_back(p);
        st.cap_path = out_path + ".cap";
    }

    if (!st.prefixes.empty()) {
        cp.cb_eval = cap_cb;
        cp.cb_eval_user_data = &st;
    }

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    if (!st.prefixes.empty()) {
        st.out.open(st.cap_path, std::ios::binary);
        st.active = true;
    }

    const llama_vocab * vocab_ptr = vocab;
    auto greedy = [&](const float * lg) {
        int top = 0;
        for (int i = 1; i < llama_vocab_n_tokens(vocab_ptr); i++) {
            if (lg[i] > lg[top]) top = i;
        }
        return top;
    };

    llama_batch batch = llama_batch_get_one(toks.data(), n_prompt);
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
    st.active = false;

    const float * logits = llama_get_logits_ith(ctx, n_prompt - 1);
    int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<float> last_logits(logits, logits + n_vocab);

    // greedy decode: step 0's logits already in hand
    printf("greedy:");
    char pbuf[128];
    for (int step = 0; step < n_predict; step++) {
        int tok = greedy(last_logits.data());
        printf(" %d", tok);
        int n = llama_token_to_piece(vocab, tok, pbuf, sizeof(pbuf), 0, true);
        if (n > 0) printf(" '%.*s'", n, pbuf);
        printf("\n");
        if (llama_vocab_is_eog(vocab, tok) || step + 1 >= n_predict) break;
        llama_batch nb = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, nb)) { fprintf(stderr, "decode failed at step %d\n", step + 1); return 1; }
        const float * lg = llama_get_logits_ith(ctx, 0);
        last_logits.assign(lg, lg + n_vocab);
    }

    std::ofstream f(out_path, std::ios::binary);
    f.write((const char *) last_logits.data(), n_vocab * sizeof(float));
    f.close();

    int top = greedy(last_logits.data());
    printf("n_vocab=%d top1=%d logit=%.4f\n", n_vocab, top, last_logits[top]);
    if (!st.prefixes.empty()) printf("capture: %s\n", st.cap_path.c_str());
    return 0;
}
