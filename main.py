import spacy
import pytextrank
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from rouge_score import rouge_scorer
from bert_score import score as bert_score

EXTRACTION_BASED_SUMMARY_NAIVE_SCORING = 0x01
EXTRACTION_BASED_SUMMARY_FIRST_LAST_SCORING = 0x02
EXTRACTION_BASED_SUMMARY_TF_IDF_SCORING = 0x04
EXTRACTION_BASED_SUMMARY_TEXT_RANK_SCORING = 0x08
EXTRACTION_BASED_SUMMARY = 0x0F

ABSTRACT_SUMMARY_T5 = 0x10
ABSTRACT_SUMMARY_PEGASUS = 0x20
ABSTRACT_SUMMARY = 0xF0


def extraction_based_summarize_naive_scoring(doc):
    sentence_scores = {}

    for i, sent in enumerate(doc.sents):
        score = 0
        for token in sent:
            if token.pos_ in ["NOUN", "ADJ", "VERB"]:
                score += 1
        score /= len(sent)
        sentence_scores[i] = score
    return sentence_scores


def extraction_based_summarize_tfidf_sentence_scoring(doc):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([sent.text for sent in doc.sents])
    sentence_scores = X.sum(axis=1)
    max_score = sentence_scores.flatten().max()
    min_score = sentence_scores.flatten().min()
    sorted_indices = sentence_scores.flatten().argsort()

    sent_score_map = {}
    for sent_id in sorted_indices[0, :].tolist()[0]:
        sent_score_map[sent_id] = (sentence_scores[sent_id] - min_score) / (max_score - min_score)

    return sent_score_map


def extraction_based_summarize_text_rank_sentence_scoring(doc):
    doc_sents_list = list(doc.sents)
    sentence_scores = {}

    for j, sent_tr in enumerate(doc._.textrank.summary(limit_sentences=len(doc_sents_list))):
        sent_id = 0
        for i, sent in enumerate(doc_sents_list):
            if sent.text == sent_tr.text:
                sent_id = i
                break
        score = float(j) / float(len(doc_sents_list))
        sentence_scores[sent_id] = score

    return sentence_scores


def add_base(sents_map, base):
    for sent_id, score in sents_map.items():
        sents_map[sent_id] = score + base
    return sents_map


def extraction_based_summarize(doc, method, num_sentences):
    doc_sents_list = list(doc.sents)

    use_naive_scoring = bool((method & EXTRACTION_BASED_SUMMARY_NAIVE_SCORING) > 0)
    use_first_last_scoring = bool((method & EXTRACTION_BASED_SUMMARY_FIRST_LAST_SCORING) > 0)
    use_tf_idf_scoring = bool((method & EXTRACTION_BASED_SUMMARY_TF_IDF_SCORING) > 0)
    use_text_rank_scoring = bool((method & EXTRACTION_BASED_SUMMARY_TEXT_RANK_SCORING) > 0)

    first_sent = doc_sents_list[0]
    last_sent = doc_sents_list[-1]
    if use_first_last_scoring and num_sentences == 1:
        return first_sent
    if use_first_last_scoring and num_sentences == 2:
        return str(first_sent).strip() + " " + str(last_sent).strip()

    if use_first_last_scoring:
        num_sentences -= 2

    sent_id_score_maps = []

    if use_naive_scoring:
        naive_sents_map = extraction_based_summarize_naive_scoring(doc)
        sent_id_score_maps.append(add_base(naive_sents_map, 0.2))

    if use_tf_idf_scoring:
        tfidf_sents_map = extraction_based_summarize_tfidf_sentence_scoring(doc)
        sent_id_score_maps.append(add_base(tfidf_sents_map, 0.4))

    if use_text_rank_scoring:
        text_rank_sents_map = extraction_based_summarize_text_rank_sentence_scoring(doc)
        sent_id_score_maps.append(add_base(text_rank_sents_map, 0.7))

    final_map = {}
    for sent_map in sent_id_score_maps:
        for sent_id, score in sent_map.items():
            if sent_id not in final_map:
                final_map[sent_id] = score
            else:
                final_map[sent_id] += score

    summary_sentences = []
    sorted_final_map_items = sorted(final_map.items(), key=lambda x: x[1], reverse=True)
    for sent_id, score in sorted_final_map_items:
        summary_sentences.append((sent_id, doc_sents_list[sent_id]))
    summary_sentences = summary_sentences[:num_sentences]
    summary_sentences = sorted(summary_sentences, key=lambda x: x[0], reverse=True)
    summary = " ".join([str(sent).strip() for _, sent in summary_sentences])

    if use_first_last_scoring:
        return str(first_sent).strip() + " " + summary + " " + str(last_sent).strip()
    else:
        return summary


def abstractive_summarization(text, method):
    use_t5 = bool((method & ABSTRACT_SUMMARY_T5) > 0)
    use_pegasus = bool((method & ABSTRACT_SUMMARY_PEGASUS) > 0)

    if use_pegasus:
        model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        summary_ids = model.generate(input_ids, min_length=28, max_length=28)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    if use_t5:
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        input_ids = tokenizer.encode(text, return_tensors='pt')
        summary_ids = model.generate(input_ids, min_length=30, max_length=120)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return "Uknown abstract summary method"


def evaluate_summary(generated_summary, reference_summary):
    """
    Ocena jakości podsumowania za pomocą metryk ROUGE i BERTScore.
    """
    # ROUGE
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(reference_summary, generated_summary)

    # BERTScore
    P, R, F1 = bert_score([generated_summary], [reference_summary], lang="en")

    evaluation = {
        'ROUGE': {k: v.fmeasure for k, v in rouge_scores.items()},
        'BERTScore': {
            'Precision': P.mean().item(),
            'Recall': R.mean().item(),
            'F1': F1.mean().item()
        }
    }
    return evaluation


def hex(value):
    try:
        return int(value, 16)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Path to model used for text information extraction using spaCy.")
    parser.add_argument('--text_path', type=str, help="Path to file containing text to summarize.")
    parser.add_argument('--method', type=hex, help="Method used to summarize text.\n\
            \tEXTRACTION_BASED_SUMMARY_NAIVE_SCORING=0x01,\n\
            \tEXTRACTION_BASED_SUMMARY_FIRST_LAST_SCORING=0x02,\n\
            \tEXTRACTION_BASED_SUMMARY_TF_IDF_SCORING=0x04,\n\
            \tEXTRACTION_BASED_SUMMARY_TEXT_RANK_SCORING=0x08,\n\
            \tABSTRACT_SUMMARY=0x10")
    parser.add_argument('--num_sentences', type=int, help="Number of sentences in the summary.")
    parser.add_argument('-rs', '--reference_path', type=str, help="(Optional) Path to file containing reference summary.")
    args = parser.parse_args()

    with open(args.text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    with open(args.reference_path, 'r', encoding='utf-8') as f:
        reference_summary = f.read()

    if (args.method & EXTRACTION_BASED_SUMMARY) > 0:
        nlp = spacy.load(args.model)
        nlp.add_pipe("textrank")
        doc = nlp(text)
        summary = extraction_based_summarize(doc, (args.method & EXTRACTION_BASED_SUMMARY), args.num_sentences)
        print(summary)
        if args.reference_path:
            eval_extraction_based = evaluate_summary(summary, reference_summary)
            print(eval_extraction_based)

    if (args.method & ABSTRACT_SUMMARY) > 0:
        abstractive_summary = abstractive_summarization(text, (args.method & ABSTRACT_SUMMARY))
        print(abstractive_summary)
        if args.reference_path:
            eval_abstractive = evaluate_summary(abstractive_summary, reference_summary)
            print(eval_abstractive)
