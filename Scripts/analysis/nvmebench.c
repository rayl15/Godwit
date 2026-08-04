// Random-read benchmark shaped like MoE expert streaming:
// N threads issuing pread() of a fixed blob size at random blob-aligned offsets.
// F_NOCACHE defeats the page cache so we measure the device, not RAM.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#include <errno.h>

static int      g_fd;
static size_t   g_blob;
static long     g_blobs;
static int      g_reads_per_thread;
static unsigned g_seed_base;

static void *worker(void *arg) {
    long id = (long)arg;
    unsigned seed = g_seed_base + (unsigned)id * 7919u;
    void *buf = NULL;
    if (posix_memalign(&buf, 16384, g_blob) != 0) return NULL;

    for (int i = 0; i < g_reads_per_thread; i++) {
        long blob = (long)(rand_r(&seed) % g_blobs);
        off_t off = (off_t)blob * (off_t)g_blob;
        size_t filled = 0;
        while (filled < g_blob) {
            ssize_t n = pread(g_fd, (char *)buf + filled, g_blob - filled, off + filled);
            if (n <= 0) { fprintf(stderr, "pread failed: %s\n", strerror(errno)); exit(1); }
            filled += (size_t)n;
        }
    }
    free(buf);
    return NULL;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <file> <blob_bytes> <threads> <reads_per_thread>\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    g_blob = (size_t)atol(argv[2]);
    int threads = atoi(argv[3]);
    g_reads_per_thread = atoi(argv[4]);
    g_seed_base = 12345u;

    g_fd = open(path, O_RDONLY);
    if (g_fd < 0) { perror("open"); return 1; }
    // Bypass the unified buffer cache: we want device latency, not RAM latency.
    if (fcntl(g_fd, F_NOCACHE, 1) < 0) perror("F_NOCACHE (continuing)");
    if (fcntl(g_fd, F_RDAHEAD, 0) < 0) perror("F_RDAHEAD (continuing)");

    struct stat st;
    if (fstat(g_fd, &st) != 0) { perror("fstat"); return 1; }
    g_blobs = (long)(st.st_size / (off_t)g_blob);
    if (g_blobs < 2) { fprintf(stderr, "file too small for blob size\n"); return 1; }

    pthread_t *tid = calloc((size_t)threads, sizeof(pthread_t));
    double t0 = now_s();
    for (long i = 0; i < threads; i++) pthread_create(&tid[i], NULL, worker, (void *)i);
    for (long i = 0; i < threads; i++) pthread_join(tid[i], NULL);
    double dt = now_s() - t0;

    long total = (long)threads * g_reads_per_thread;
    double bytes = (double)total * (double)g_blob;
    printf("%2d threads | %6.1f MiB blob | %4ld reads | %6.3f s | %7.2f MiB/s | %7.2f ms/read\n",
           threads, g_blob / 1048576.0, total, dt,
           bytes / dt / 1048576.0, dt / total * 1000.0 * threads);
    close(g_fd);
    free(tid);
    return 0;
}
