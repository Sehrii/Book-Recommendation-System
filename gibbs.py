import numpy as np
import random

class LDA:
    def __init__(self, n_topics=10, alpha=0.1, beta=0.01, n_iter=1000):
        self.K = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter

    def fit(self, X):
        self.X = X
        self.D, self.V = X.shape
        print(f"Initializing LDA with {self.D} documents, {self.V} words, {self.K} topics")

        self.docs = []
        for d in range(self.D):
            doc = []
            for v in range(self.V):
                count = int(X[d, v])
                doc.extend([v] * count)
            self.docs.append(doc)
        print("Converted frequency matrix to word lists.")

        self.z = []
        self.n_dk = np.zeros((self.D, self.K), dtype=int)
        self.n_kv = np.zeros((self.K, self.V), dtype=int)
        self.n_k = np.zeros(self.K, dtype=int)

        print("Randomly assigning initial topics...")
        for d, doc in enumerate(self.docs):
            current_doc_topics = []
            for word in doc:
                topic = random.randint(0, self.K - 1)
                current_doc_topics.append(topic)
                self.n_dk[d, topic] += 1
                self.n_kv[topic, word] += 1
                self.n_k[topic] += 1
            self.z.append(current_doc_topics)
        print("Initial topic assignment complete.")

        # Gibbs Sampling
        print("Starting Gibbs sampling...")
        report_every = max(1, self.n_iter // 10)
        for it in range(self.n_iter):
            for d, doc in enumerate(self.docs):
                for i, word in enumerate(doc):
                    topic = self.z[d][i]
                    self.n_dk[d, topic] -= 1
                    self.n_kv[topic, word] -= 1
                    self.n_k[topic] -= 1

                    p_z = np.zeros(self.K)
                    for k in range(self.K):
                        term1 = (self.n_kv[k, word] + self.beta) / (self.n_k[k] + self.V * self.beta)
                        term2 = (self.n_dk[d, k] + self.alpha)
                        p_z[k] = term1 * term2

                    p_z /= np.sum(p_z)
                    new_topic = self._sample_multinomial(p_z)

                    self.z[d][i] = new_topic
                    self.n_dk[d, new_topic] += 1
                    self.n_kv[new_topic, word] += 1
                    self.n_k[new_topic] += 1

            if (it + 1) % report_every == 0 or (it + 1) == self.n_iter:
                print(f"Completed {it + 1} / {self.n_iter} iterations")

        print("Gibbs sampling finished.")

    def get_document_topic_distribution(self):
        theta = (self.n_dk + self.alpha) / (np.sum(self.n_dk, axis=1, keepdims=True) + self.K * self.alpha)
        return theta

    def _sample_multinomial(self, prob_dist):
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(prob_dist):
            cumulative += p
            if r < cumulative:
                return i
        return len(prob_dist) - 1

# ====== Main Execution ======
if __name__ == "__main__":
    print("Loading frequency matrix from freq.npy...")
    X = np.load("freq.npy")

    lda = LDA(n_topics=10, alpha=0.1, beta=0.01, n_iter=100)
    lda.fit(X)

    print("Computing document-topic matrix...")
    doc_topic_matrix = lda.get_document_topic_distribution()

    np.set_printoptions(precision=4, suppress=True)
    print("Document-Topic Matrix (books × topics):")
    print(doc_topic_matrix)

    print("Saving results to sol.npy...")
    np.save("sol.npy", doc_topic_matrix)
    print("All done!")
