import re
import gymnasium as gym
import numpy as np
from typing import List
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from sentence_transformers import SentenceTransformer
import torch
from syllapy import count
from textblob import TextBlob

class HaikuEnvironment(gym.Env):
    def __init__(self, objects, sentiment, *args, **kwargs): # It's possible to add other constraints instead of Haiku's 5 - 7 - 5 syllable requirement
        super().__init__()
        self.objects = objects
        self.sentiment = sentiment
        self.syllables_remaining = [5, 7, 5]
        self.haiku = ["", "", ""]
        self.current_line = 0
        
        self.device = torch.device("cpu")


        # Load models
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        #self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        obs_vector_size = 384  # Update to match the embedding model output size
        self.observation_space = gym.spaces.Space({
            "scene_embedding": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_vector_size,), dtype=np.float32),
            "objects_embedding": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_vector_size,), dtype=np.float32),
            "sentiment_embedding": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_vector_size,), dtype=np.float32),
            "syllables_remaining": gym.spaces.MultiDiscrete([5, 7, 5]),
        })

        # Action space remains as indices for LM suggestions
        self.action_space = gym.spaces.Discrete(self.num_suggestions)
        
    def generate_prompt(self):
        # Combine the current haiku with the objects and sentiment
        haiku_prompt = " ".join(self.haiku[:self.current_line])  # Current haiku lines
        prompt = f"Objects: {self.objects}, Sentiment: {self.sentiment}, Current Haiku: {haiku_prompt}"

        return prompt

    def step(self, action):
        prompt = f"Objects: {self.objects}, Sentiment: {self.sentiment}, Current Line: {self.haiku[self.current_line]}"
        suggestions = self.get_lm_suggestions(prompt)
        chosen_word = suggestions[action]

        # Process chosen word
        syllables = self.count_syllables(chosen_word)
        if syllables > self.syllables_remaining[self.current_line]:
            return self._get_obs(), -1, False, {}

        self.haiku[self.current_line] += chosen_word + " "
        self.syllables_remaining[self.current_line] -= syllables

        if self.syllables_remaining[self.current_line] == 0:
            self.current_line += 1

        done = self.current_line == 3
        reward = self.scorer(self.objects, self.sentiment, self.haiku) if done else 0
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.observation_space

    def get_lm_suggestions(self, prompt):
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate text with the model based on the current haiku
        outputs = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 10,  # Generate up to 10 more tokens
            num_return_sequences=1,  # Only one sequence
            do_sample=True,  # Enable sampling for diversity
            top_k=50,  # Top-k sampling for diversity
            top_p=0.95,  # Nucleus sampling for diversity
            temperature=0.7,  # Control randomness in the output
        )

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract words from the generated text, excluding the initial prompt
        words = generated_text[len(prompt):].strip().split()

        # Filter out non-alphabetic words and return the first suggestion
        suggestion = next((word for word in words if word.isalpha()), None)

        return suggestion


    def reward_function(self, haiku, target_syllables, target_sentiment, target_objects, scene_description):
        reward = 0
        
        # Checking Structure Coherence
        # This should have the highest weight because of its importance
        for i, line in enumerate(haiku):
            actual_syllables = self.count_syllables(line)  
            if actual_syllables == target_syllables[i]:
                reward += 3  
            else:
                reward -= 3  
        
        # Checking Sentiment Coherence
        # Reward is continuous - the sentiment difference
        
        haiku_sentiment = self.analyze_sentiment(haiku)  # Use sentiment analysis model (TextBlob, VADER, etc.)
        reward += haiku_sentiment - self.sentiment

        
        # Checking Object Coherence
        # Reward is the percentage of matched objects

        matched_objs = self.extract_objects(haiku,)
        reward += len(matched_objs) / len(self.objects)
        
        return reward
    
    def count_syllables(self, line):
        syllables_count = count(line)
        return syllables_count

    def analyze_sentiment(self,  haiku):
        return TextBlob("haiku").sentiment.polarity # [-1, 1] - negative to positive emotion
    
    def extract_objects(self, text, object_list):
        # Since the haiku is a list of three strings, we want to join them since in this context sentences aren't important
        
        text = " ".join(text) 
        
        text = text.lower()

        matched_objects = []

        for obj in object_list:
            if re.search(r'\b' + re.escape(obj) + r'\b', text):
                matched_objects.append(obj)

        # We'll return a list of the objects that have already been matched
        # len(matched_objects) indicates how many objects have been matched
        return matched_objects 
    

    def generate_random_scene():
        # We'll generate a random list of objects and a sentiment score
        return

    def reset(self, seed=None, options=None):
        # Seed the environment
        super().reset(seed=seed)
        np.random.seed(seed)

        # Reset the environment's state
        self.haiku = ["", "", ""]
        self.syllables_remaining = [5, 7, 5]
        self.current_line = 0

        # Return the observation and an empty info dictionary
        return self._get_obs(), {}

