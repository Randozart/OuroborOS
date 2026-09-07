/* ibv_poll_cq, ibv_post_send, ibv_post_recv are all static inlines
 * that dispatch through cq->ops / qp->context->ops.  This shim
 * compiles against the real header and exports linkable symbols. */
#include <infiniband/verbs.h>

int ouro_ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
    return ibv_poll_cq(cq, num_entries, wc);
}

int ouro_ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                       struct ibv_send_wr **bad_wr) {
    return ibv_post_send(qp, wr, bad_wr);
}

int ouro_ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                       struct ibv_recv_wr **bad_wr) {
    return ibv_post_recv(qp, wr, bad_wr);
}
