import numpy as np
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

def get_application_score(body, request):
    try:
        cover_letter = body.cover_letter
        profile_topics = body.profile_topics
        is_resume_included = body.is_resume_included
        resume_topics = body.resume_topics
        opening_description_topics = body.opening_description_topics
        organization_values_topics = body.organization_values_topics
        years_of_experience = body.years_of_experience

        miniLM_tokenizer = request.app.state.miniLM_tokenizer
        miniLM_model = request.app.state.miniLM_model

        bert_tokenizer = request.app.state.bert_tokenizer
        bert_model = request.app.state.bert_model

        if len(resume_topics)==0:
            is_resume_included = False

        cover_letter_sentiment = TextBlob(cover_letter).sentiment.polarity

        profile_topic_embeddings1 = get_embeddings(profile_topics, miniLM_model, miniLM_tokenizer)
        profile_topic_embeddings2 = get_embeddings(profile_topics, bert_model, bert_tokenizer)

        if is_resume_included:
            resume_topic_embeddings1 = get_embeddings(resume_topics, miniLM_model, miniLM_tokenizer)
            resume_topic_embeddings2 = get_embeddings(resume_topics, bert_model, bert_tokenizer)

        opening_description_topic_embeddings1 = get_embeddings(opening_description_topics, miniLM_model, miniLM_tokenizer)
        opening_description_topic_embeddings2 = get_embeddings(opening_description_topics, bert_model, bert_tokenizer)

        organization_values_topic_embeddings1 = get_embeddings(organization_values_topics, miniLM_model, miniLM_tokenizer)
        organization_values_topic_embeddings2 = get_embeddings(organization_values_topics, bert_model, bert_tokenizer)

        profile_similarity_score = get_final_emb_score(
            get_emb_score(profile_topic_embeddings1, opening_description_topic_embeddings1, organization_values_topic_embeddings1, 0.4),
            get_emb_score(profile_topic_embeddings2, opening_description_topic_embeddings2, organization_values_topic_embeddings2, 0.75)
        )
        if is_resume_included:
            resume_similarity_score = get_final_emb_score(
            get_emb_score(resume_topic_embeddings1, opening_description_topic_embeddings1, organization_values_topic_embeddings1, 0.4),
            get_emb_score(resume_topic_embeddings2, opening_description_topic_embeddings2, organization_values_topic_embeddings2, 0.75)
        )
        else:
            resume_similarity_score = 0

        a = len(profile_topics)
        b = len(resume_topics)
        m = len(opening_description_topics)
        n = len(organization_values_topics)

        dynamic_max_profile_score = a
        dynamic_max_resume_score = b

        if dynamic_max_profile_score > 5:
            profile_score_cap = 5
        else:
            profile_score_cap = dynamic_max_profile_score

        if dynamic_max_resume_score > 5:
            resume_score_cap = 5
        else:
            resume_score_cap = dynamic_max_resume_score
            
        work_ex_score_cap = 3

        sentiment_score = cover_letter_sentiment
        profile_score = profile_similarity_score
        resume_score = resume_similarity_score

        if is_resume_included:
            if profile_score > dynamic_max_profile_score*0.1 and resume_score > dynamic_max_resume_score*0.1:
                work_ex_score = years_of_experience
            else:
                work_ex_score = 0
        else:
            if profile_score > dynamic_max_profile_score*0.1:
                work_ex_score = years_of_experience
            else:
                work_ex_score = 0

        if is_resume_included:
            weight_sentiment = 0.10
            weight_profile = 0.35
            weight_resume = 0.40
            weight_work_ex = 0.15
        else:
            weight_sentiment = 0.2
            weight_profile = 0.6
            weight_resume = 0
            weight_work_ex = 0.2

        weighted_sentiment_score = sentiment_score * weight_sentiment
        weighted_profile_score = min(profile_score, profile_score_cap) * weight_profile
        weighted_resume_score = min(resume_score, resume_score_cap) * weight_resume
        weighted_work_ex_score = min(work_ex_score, work_ex_score_cap) * weight_work_ex

        overall_fit_score = weighted_sentiment_score + weighted_profile_score + weighted_resume_score + weighted_work_ex_score

        max_score = 1*0.1 + profile_score_cap*0.35 + resume_score_cap*0.4 + 3*0.15 # Maximum possible weighted score
        min_score = -0.1  # Minimum possible weighted score

        overall_fit_score_normalized = (overall_fit_score - min_score) / (max_score - min_score)

        return {
            'score': overall_fit_score_normalized
        }
    except Exception as e:
        print(e)
        return {
            'score': -1
        }
    

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(sentences, model, tokenizer):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings

def get_similarity_score(emb1, emb2):
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)
    
    # Calculate cosine similarity
    return cosine_similarity(emb1, emb2)[0][0]

def get_emb_score(embeddings, opening_description_topic_embeddings, organization_values_topic_embeddings, treshold):
    score = 0

    for i in embeddings:
        for j in np.concatenate((opening_description_topic_embeddings, organization_values_topic_embeddings)):
            similarity_score = get_similarity_score(i,j)
            if similarity_score>treshold:
                score+=similarity_score

    return score

def get_final_emb_score(emb_score1, emb_score2):
    return emb_score1*0.9+emb_score2*0.1