import multiprocessing
import re
import regex
from collections import Counter
form typing import list, dict, tuple

GPT2_TOKENIZER_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init  (self, config):
        self.config = config
        self.vocab: dict[bytes, int] = {}
        self.merges: list[tuple[bytes, bytes], int] = []

        # 用于解码的辅助结构
        self.inverse_vocab = {}

        self.next_token_id = 256
        
        self.special_tokens: set[bytes] = set()
        self.special_token_pattern: re.Pattern | None = None

    def init_bytes_to_unicode() -> dict[int, str]:
        '''
        字节到Unicode字符的映射
        '''
        return {i: chr(i) for i in range(256)}

    def load_and_sample_data(self, file_path: str, sample_size: int = 22000, special_token: str = "<|endoftext|>") -> str:
        '''
        内存映射方式加载并采样文档
        '''
        try:
            with open(file_path, "r+", encoding='utf-8', errors='ignore') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    documents = []
                    start = 0
                    while start < len(mm):
                        end = mm.find(special_token.encode('utf-8'), start)
                        # 没有找到
                        if end == -1:
                            doc = mm[start:].decode('utf-8', errors='replace').strip()
                            if doc:
                                documents.append(doc)
                            break
                        # 找到
                        doc = mm[start:end].decode('utf-8', errors='replace').strip()
                        if doc:
                            documents.append(doc)
                        start = end + len(special_token)
                    # 如果文档数小于采样数，随机采样
                    if len(documents) < sample_size:
                        documents = random.sample(documents, sample_size)
                    # 使用特殊标记连接文档
                    return special_token.join(documents)
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return ""

    def pre_tokenize_document(doc: str, bytes_to_unicode: dict[int, str]) -> Counter[tuple[bytes, ...]]:
        '''
        使用GPT-2的regex将文本预分词，编码成UTF-8，返回Counter
        '''
        pattern = re.compile(GPT2_TOKENIZER_REGEX, re.UNICODE)
        tokens_counter = Counter()

        for match in regex.finditer(pattern, doc):
            token = match.group()
            token_bytes = token.encode('utf-8')
            byte_tuple = tuple(bytes[b] for b in token_bytes)
            tokens_counter[byte_tuple] += 1
        return tokens_counter

    def parallel_pre_tokenize(self, documents: list[str], num_processes: int, bytes_to_unicode: dict[int, str]) -> list[Counter[tuple[bytes, ...]]]:
        '''
        并行预分词
        '''
        if num_processes <= 1:
            return [self.pre_tokenize_document(doc, bytes_to_unicode) for doc in documents]
        pass


    def count_pair_frequencies(self, tokens_counter: Counter[tuple[bytes]]) -> dict[tuple[bytes, bytes], int]:
        counts = defaultdict(int)
        for token, count in tokens_counter.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                counts[pair] += count
        return counts

    def merge_tokens(self, tokens_counter: Counter[tuple[bytes]], match1: bytes, match2: bytes) -> Counter[tuple[bytes]]:
        new_counter = Counter()
        for word, freq in tokens_counter.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == match1 and word[i+1] == match2:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_counter[tuple(new_word)] += freq
        return new_counter

        

    def train(self, 
        text: str,
        vocab_size: int = 1000,
        special_tokens: list[str] = ["<|endoftext|>"],
        num_processes: int = 8,
        sample_size: int = 22000,
        **kwargs
    ):
        if self.config["vocab_size"] < 256:
            raise ValueError("词汇表大小必须至少为 256 以覆盖所有字节。")
        # 1.初始化
        bytes_to_unicode = self.init_bytes_to_unicode()
        unicode_to_bytes = {v: k for k, v in bytes_to_unicode.items()}
        self.vocab = {i: bytes([i]) for i in range(256)}\
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_token_id = 256

        # 2.添加特殊标记
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            self.vocab[self.next_token_id] = token_bytes
            self.inverse_vocab[token_bytes] = self.next_token_id
            self.next_token_id += 1

        # 3.加载并采样文档
        text = self.load_and_sample_data(text, sample_size, special_tokens[0])

        # 4.分割文档
        escaped_tokens = [re.escape(token) for token in special_tokens]
        split_pattern = re.compile("|".join(escaped_tokens))
        documents = split_pattern.split(text)

        # 5.预分词
        sequences = self.pre_tokenize_document(documents, bytes_to_unicode)
        # sequences = self.parallel_pre_tokenize(documents, num_processes, bytes_to_unicode)
        print(f"预分词完成，总序列数: {len(sequences)}")

        encoded_text = text.encode('utf-8')
        # 使用GPT-2的regex将文本分割成单词
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)


        word_splits = [[]]
        

    def tokenize(self, text):
        pass

    def encode(self, text):
        pass

    def decode(self, token_ids):
        pass
        

if __name__ == "__main__":
    config = {
        "vocab_size": 1000,
        "special_tokens": ["<|endoftext|>", "<pad>", "<unk>"],
        "num_processes": 8,
        "sample_size": 22000,
    }

    # 数据集
    train_path = "TinyStories-train.txt"
    valid_path = "TinyStories-valid.txt"
    if not os.path.exists(train_path):
        print("训练数据集不存在，请下载")
    if not os.path.exists(valid_path):
        print("验证数据集不存在，请下载")

    # 训练
    print("开始训练")
    start_time = time.time()
    tokenizer = BPETokenizer(config)
    tokenizer.train(train_path)
    end_time = time.time()
    print(f"训练耗时: {end_time - start_time:.2f}秒")