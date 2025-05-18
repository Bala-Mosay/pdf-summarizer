import networkx as nx
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from rouge_score import rouge_scorer
import torch

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

class HybridSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.stop_words = set(stopwords.words("english"))
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def create_word_graph(self, text):
        # Tokenize and filter words
        words = nltk.word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]

        # Use POS tagging to retain content words
        pos_tags = nltk.pos_tag(filtered_words)
        content_words = [word for word, pos in pos_tags if pos.startswith(("NN", "VB", "JJ", "RB"))]

        # Build the graph
        graph = nx.DiGraph()
        for i, word in enumerate(content_words):
            for j in range(i + 1, min(i + 6, len(content_words))):
                graph.add_edge(word, content_words[j], weight=1.0 / (j - i))

        return graph

    def extract_key_sentences(self, graph, original_text, num_sentences=3):
        # Calculate PageRank scores
        scores = nx.pagerank(graph)
        ranked_words = sorted(scores, key=scores.get, reverse=True)

        # Match ranked words to original sentences
        sentences = nltk.sent_tokenize(original_text)
        selected_sentences = []
        for word in ranked_words:
            for sentence in sentences:
                if word in sentence and sentence not in selected_sentences:
                    selected_sentences.append(sentence)
                    break
            if len(selected_sentences) >= num_sentences:
                break

        return selected_sentences

    def generate_summary(self, sentences):
        # Combine the extracted sentences
        input_text = " ".join(sentences)
        inputs = self.tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate summary with repetition control
        summary_ids = self.model.generate(
            inputs,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            repetition_penalty=2.5,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    def evaluate_summary(self, reference, generated_summary):
        # Calculate ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, generated_summary)
        return rouge_scores

    def summarize_with_evaluation(self, text, reference_summary=None):
        # Step 1: Graph-Based Preprocessing
        graph = self.create_word_graph(text)
        key_sentences = self.extract_key_sentences(graph, text)

        # Step 2: Transformer-Based Summarization
        generated_summary = self.generate_summary(key_sentences)

        # Step 3: Evaluate Summary (if reference provided)
        if reference_summary:
            rouge_scores = self.evaluate_summary(reference_summary, generated_summary)
            return {
                'summary': generated_summary,
                'key_sentences': key_sentences,
                'rouge_scores': rouge_scores
            }
        
        return {
            'summary': generated_summary,
            'key_sentences': key_sentences
        }

def main():
    # Example Usage
    text = """
    Summarization is the process of condensing a large amount of information into a more concise and coherent form while preserving its key points and overall meaning. 
    It is an essential technique in fields like natural language processing (NLP) and is widely used for creating abstracts, news summaries, and document insights. 
    Summarization can be classified into two main types: extractive and abstractive. Extractive summarization involves selecting and combining important sentences or phrases directly from the source text, while abstractive summarization generates new sentences that capture the core ideas. 
    This technique helps in quickly understanding large volumes of text and improving information retrieval.
    """

    # Optional reference summary for evaluation
    reference_summary = "Summarization condenses information while preserving key points. It has two types: extractive and abstractive, both used in NLP for creating concise text representations."

    # Create summarizer
    summarizer = HybridSummarizer()

    # Summarize with evaluation
    result = summarizer.summarize_with_evaluation(text, reference_summary)

    print("Key Sentences:")
    for sentence in result['key_sentences']:
        print(f"- {sentence}")

    print("\nGenerated Summary:")
    print(result['summary'])

    if 'rouge_scores' in result:
        print("\nROUGE Scores:")
        rouge_scores = result['rouge_scores']
        print(f"ROUGE-1: {rouge_scores['rouge1']}")
        print(f"ROUGE-2: {rouge_scores['rouge2']}")
        print(f"ROUGE-L: {rouge_scores['rougeL']}")

if __name__ == "__main__":
    main()
