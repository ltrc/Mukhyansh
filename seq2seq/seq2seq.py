# Imports 

import random
import codecs
import math
import time
import sys
import os.path
import random
import json
import argparse
from operator import itemgetter
import pandas as pd
import numpy as np
from numpy import inf
from tqdm import tqdm

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , TimeDistributed , LSTM,GRU, Embedding, Lambda, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence

from sacrebleu.metrics import BLEU
from multilingual_rouge_score import rouge_scorer
from bpemb import BPEmb
import fasttext

 
class HeadlineGeneration():
    def __init__(self, vocab_df, use_bpe, emb_model , scorer, rnn_type):
        self.vocab_df = vocab_df
        self.use_bpe = use_bpe
        self.emb_model = emb_model
        self.scorer = scorer
        self.word2vec = None
        self.idx2word = {}
        self.word2idx = {}
        self.rnn_type = rnn_type
        
        # initalize end of sentence, empty and unk tokens
        self.word2idx['<empty>'] = empty_tag_location
        self.word2idx['<eos>'] = eos_tag_location
        self.word2idx['<unk>'] = unknown_tag_location
        self.idx2word[empty_tag_location] = '<empty>'
        self.idx2word[eos_tag_location] = '<eos>'
        self.idx2word[unknown_tag_location] = '<unk>'
    
    def read_word_embedding(self):
        """
            This function will create a word2vec matrix for a given vocabulary of words.
        """
        idx = 3
        temp_word2vec_dict = {}
        # <empty>, <eos> tag replaced by word2vec learning
        # create random dimensional vector for empty, eos and unk tokens
        temp_word2vec_dict['<empty>'] = [float(i) for i in np.random.rand(embedding_dimension, 1)]
        temp_word2vec_dict['<eos>'] = [float(i) for i in np.random.rand(embedding_dimension, 1)]
        temp_word2vec_dict['<unk>'] = [float(i) for i in np.random.rand(embedding_dimension, 1)]
        # loading the embedding model
        model = self.emb_model
        if use_bpe=="True":
            words = self.vocab_df["tokens"]
            word_indexes = self.vocab_df["Index"]
            for word,word_index in zip(words,word_indexes):
                vector = model.vectors[word_index]
                temp_word2vec_dict[idx] = vector
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx = idx + 1
                if idx % 10000 == 0:
                    print ("Loading BPE embeddings...", idx)
        else:
            V = vocab_df["tokens"][:40000]
            for word in V:
                vector = model[word]
                temp_word2vec_dict[idx] = vector
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx = idx + 1
                if idx % 10000 == 0:
                    print ("Loading FastText embeddings...", idx)
            

        length_vocab = len(temp_word2vec_dict)
        shape = (length_vocab, embedding_dimension)
        # print(shape)
        # faster initlization and random for <empty> and <eos> tag
        self.word2vec = np.random.uniform(low=-1, high=1, size=shape)
        for i in tqdm(range(length_vocab)):
            if i in temp_word2vec_dict:
                self.word2vec[i, :] = temp_word2vec_dict[i]
    
    def output_shape_simple_context_layer(self,input_shape):
        """
        Take input shape tuple and return tuple for output shape
        Output shape size for simple context layer =
        remaining part after activatoion calculation fron input layers avg. +
        remaining part after activatoion calculation fron current hidden layers avg.
        
        that is 2 * (rnn_size - activation_rnn_size))
        
        input_shape[0] = batch_size remains as it is
        max_len_head = heading max length allowed
        """
        return (input_shape[0], max_len_head , 2 * (rnn_size - activation_rnn_size))
    
    def simple_context(self, X, mask):
        """
        Simple context calculation layer logic
        X = (batch_size, time_steps, units)
        time_steps are nothing but number of words in our case.
        """
        # segregrate heading and desc
        desc, head = X[:, :max_len_desc, :], X[:, max_len_desc:, :]
        # segregrate activation and context part
        head_activations, head_words = head[:, :, :activation_rnn_size], head[:, :, activation_rnn_size:]
        desc_activations, desc_words = desc[:, :, :activation_rnn_size], desc[:, :, activation_rnn_size:]
        
        # p=(bacth_size, length_desc_words, rnn_units)
        # q=(bacth_size, length_headline_words, rnn_units)
        # K.dot(p,q) = (bacth_size, length_desc_words,length_headline_words)
        activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2, 2))
        
        # make sure we dont use description words that are masked out
        activation_energies = activation_energies + -1e20 * K.expand_dims(1. - K.cast(mask[:, :max_len_desc], 'float32'), 1)
        activation_energies = K.reshape(activation_energies, (-1, max_len_desc))
        activation_weights = K.softmax(activation_energies)
        activation_weights = K.reshape(activation_weights, (-1, max_len_head, max_len_desc))
        
        # for every head word compute weighted average of desc words
        desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2, 1))
        return K.concatenate((desc_avg_word, head_words))
        
    def create_model(self,):
        """
        RNN model creation
        Layers include Embedding Layer, 3 LSTM stacked,
        Simple Context layer (manually defined),
        Time Distributed Layer
        """
        length_vocab, embedding_size = self.word2vec.shape
        print ("shape of word2vec matrix ", self.word2vec.shape)

        model = Sequential()

        # TODO: look at mask zero flag
        model.add(
                Embedding(
                        length_vocab, embedding_size,
                        input_length=max_length,
                        weights=[self.word2vec], mask_zero=True,
                        name='embedding_layer'
                )
        )

        for i in range(rnn_layers):
            if self.rnn_type=="gru":
                rnn = GRU(rnn_size, return_sequences=True,
                    name= 'GRU_layer_%d' % (i + 1)
                )
            else:
                rnn = LSTM(rnn_size, return_sequences=True,
                    name= 'LSTM_layer_%d' % (i + 1)
                )

            model.add(rnn)
            # No drop out added ! 
        

        model.add(Lambda(self.simple_context,
                     mask = lambda inputs, mask: mask[:, max_len_desc:],
                     output_shape = self.output_shape_simple_context_layer,
                     name = 'simple_context_layer'))

        vocab_size = self.word2vec.shape[0]
        model.add(TimeDistributed(Dense(vocab_size,
                                name='time_distributed_layer')))
        
        model.add(Activation('softmax', name='activation_layer'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        K.set_value(model.optimizer.lr, np.float32(learning_rate))
        print (model.summary())
        return model
    
    def padding(self, list_idx, curr_max_length, is_left):
        """
        padds with <empty> tag in left side
        """
        if len(list_idx) >= curr_max_length:
            return list_idx
        number_of_empty_fill = curr_max_length - len(list_idx)
        if is_left:
            return [empty_tag_location, ] * number_of_empty_fill + list_idx
        else: 
            return list_idx + [empty_tag_location, ] * number_of_empty_fill

    def headline2idx(self, list_idx, curr_max_length, is_input):
        """
        if space add <eos> tag in input case, input size = curr_max_length-1
        always add <eos> tag in predication case, size = curr_max_length
        always right pad
        """
        if is_input:
            if len(list_idx) >= curr_max_length - 1:
                return list_idx[:curr_max_length - 1]
            else:
                # space remaning add eos and empty tags
                list_idx = list_idx + [eos_tag_location, ]
                return self.padding(list_idx, curr_max_length - 1, False)
        else:
            # always add <eos>
            if len(list_idx) == curr_max_length:
                list_idx[-1] = eos_tag_location
                return list_idx
            else:
                # space remaning add eos and empty tags
                list_idx = list_idx + [eos_tag_location, ]
                return self.padding(list_idx, curr_max_length, False)
    
    def desc2idx(self, list_idx, curr_max_length):
        """
        always left pad and eos tag to end
        """
        #====== REVERSE THE DESC IDS ========
        list_idx.reverse()
        # padding to the left remain same and 
        # eos tag position also remain same,
        # just description flipped
        #===================================
        # desc padded left
        list_idx = self.padding(list_idx, curr_max_length, True)
        # eos tag add
        list_idx = list_idx + [eos_tag_location, ]
        return list_idx
    
    def sentence2idx(self, sentence, is_headline, curr_max_length, is_input=True):
        """
        given a sentence convert it to its ids
        "I like India" => [12, 51, 102]
        if words not present in vocab ignore them
        is_input is only for headlines
        """
        list_idx = []
        if self.use_bpe:            
            tokens= self.emb_model.encode(sentence)
        else:
            tokens= sentence.split(" ")
        count = 0
        for each_token in tokens:
            if each_token in self.word2idx:
                list_idx.append(self.word2idx[each_token])
            else:
                #append unk token as original word not present in word2vec
                list_idx.append(self.word2idx['<unk>'])
            count = count + 1
            if count >= curr_max_length:
                break

        if is_headline:
            return self.headline2idx(list_idx, curr_max_length, is_input)
        else:
            return self.desc2idx(list_idx, curr_max_length)
    
    def flip_words_randomly(self, description_headline_data, number_words_to_replace, model):
        """
        Given actual data i.e. description + eos + headline + eos
        1. It predicts news headline (model try to predict, sort of training phase)
        2. From actual headline, replace some of the words,
        with most probable predication word at that location
        3.return description + eos + headline(randomly some replaced words) + eos
        (take care of eof and empty should not get replaced)
        """
        if number_words_to_replace <= 0 or model == None:
            return description_headline_data

        # check all descrption ends with <eos> tag else throw error
        assert np.all(description_headline_data[:, max_len_desc] == eos_tag_location)

        batch_size = len(description_headline_data)
        predicated_headline_word_idx = model.predict(description_headline_data, verbose=0, batch_size=batch_size)
        copy_data = description_headline_data.copy()
        for idx in range(batch_size):
            # description = 0 ... max_len_desc-1
            # <eos> = max_len_desc
            # headline = max_len_desc + 1 ...
            random_flip_pos = sorted(random.sample(range(max_len_desc + 1, max_length), number_words_to_replace))
            for replace_idx in random_flip_pos:
                # Don't replace <eos> and <empty> tag
                if (description_headline_data[idx, replace_idx] == empty_tag_location or
                description_headline_data[idx, replace_idx] == eos_tag_location):
                    continue

                # replace_idx offset moving as predication doesnot have desc
                new_id = replace_idx - (max_len_desc + 1)
                prob_words = predicated_headline_word_idx[idx, new_id]
                word_idx = prob_words.argmax()
                # dont replace by empty location or eos tag location
                if word_idx == empty_tag_location or word_idx == eos_tag_location:
                    continue
                copy_data[idx, replace_idx] = word_idx
        return copy_data
    
    def convert_inputs(self, descriptions, headlines,number_words_to_replace, model,is_training):
        """
        convert input to suitable format
        1.Left pad descriptions with <empty> tag
        2.Add <eos> tag
        3.Right padding with <empty> tag after (desc+headline)
        4.input headline doesnot contain <eos> tag
        5.expected/predicated headline contain <eos> tag
        6.One hot endcoing for expected output
        """
        # length of headlines and descriptions should be equal
        assert len(descriptions) == len(headlines)

        X, y = [], []
        for each_desc, each_headline in zip(descriptions, headlines):
            input_headline_idx = self.sentence2idx(each_headline, True, max_len_head, True)
            predicted_headline_idx = self.sentence2idx(each_headline, True, max_len_head, False)
            desc_idx = self.sentence2idx(each_desc, False, max_len_desc)
            
            # assert size checks
            assert len(input_headline_idx) == max_len_head - 1
            assert len(predicted_headline_idx) == max_len_head
            assert len(desc_idx) == max_len_desc + 1

            X.append(desc_idx + input_headline_idx)
            y.append(predicted_headline_idx)
            
        X, y = np.array(X), np.array(y)
        if is_training:
            X = self.flip_words_randomly(X, number_words_to_replace, model)
            # One hot encoding of y
            vocab_size = self.word2vec.shape[0]
            length_of_data = len(headlines)
            Y = np.zeros((length_of_data, max_len_head, vocab_size))
            for i, each_y in enumerate(y):
                Y[i, :, :] = to_categorical(each_y, vocab_size)
            #check equal lengths
            assert len(X)==len(Y)
            return X, Y
        else:
            #Testing doesnot require OHE form of headline, flipping also not required
            #Because BLUE score require words and not OHE form to check accuracy
            return X,headlines        
        
    def large_file_reading_generator(self,data):
        """
        Read a large file line by line.
        """
        for each_line in data:
            yield each_line
            
    def data_generator(self, file_name,batch_size,number_words_to_replace,model,is_training=True):
        """
        Read a JSONL file in chunks and return a chunk of data to train on.
        """
        with open(file_name, 'r', encoding='utf-8') as file_pointer:
            data = [json.loads(line) for line in file_pointer]

            if is_training:
                # Shuffle the dataset.
                random.shuffle(data)

        headlines_data = [item['title'] for item in data]
        descs_data = [item['text'] for item in data]

        headline_iterator = self.large_file_reading_generator(headlines_data)
        descs_iterator = self.large_file_reading_generator(descs_data)

        while True:
            X, y = [], []
            for i in range(batch_size):
                heads_line = next(headline_iterator)
                descs_line = next(descs_iterator)
                y.append(heads_line)
                X.append(descs_line)
            yield self.convert_inputs(X, y, number_words_to_replace, model, is_training)
    
    def OHE_to_indexes(self,y_val):
        """
        reverse of OHE 
        OHE => indexes
        e.g. [[0,0,1],[1,0,0]] => [2,0]
        """
        list_of_headline = []
        for each_headline in y_val:
            list_of_word_indexes = np.where(np.array(each_headline)==1)[1]
            list_of_headline.append(list(list_of_word_indexes))
        return list_of_headline
    
    def indexes_to_words(self, list_of_headline):
        """
        indexes => words (for BLUE Score)
        e.g. [2,0] => ["I","am"] (idx2word defined dictionary used)
        """
        list_of_word_headline = []
        for each_headline in list_of_headline:
            each_headline_words = []
            for each_word in each_headline:
                #Dont include <eos> and <empty> tags
                if each_word in (empty_tag_location, eos_tag_location, unknown_tag_location):
                    continue
                each_headline_words.append(self.idx2word[each_word])
            list_of_word_headline.append(each_headline_words)            
        return list_of_word_headline
    
    def blue_score_text(self, y_actual,y_predicated):
        #check length equal
        assert len(y_actual) ==  len(y_predicated)
        #list of healine .. each headline has words
        no_of_news = len(y_actual)
    
        ###### sacre blue ######
        sacre_blue_score=0
        bleu = BLEU(smooth_method='add-k',effective_order=True)
        ########################

        r_list = []

        for i in range(no_of_news):
            reference = y_actual[i]
            hypothesis = y_predicated[i]
            ref = " ".join(reference)
            hyp = " ".join(hypothesis)
                
            ###### multilingual ROUGE #######
            scores = self.scorer.score(ref, hyp) # ref and hyp must be in string format only
            r1_f=scores['rouge1'][2] # index 2 corresponds to f-1 score, 0-precsion,1-recall
            r2_f=scores['rouge2'][2]
            rL_f=scores['rougeL'][2]
            # converting it to old rouge output format to avoid code changes in the later sections
            r_score={'rouge-1':{'f':r1_f},
                    'rouge-2':{'f':r2_f},
                    'rouge-l':{'f':rL_f}}
            r_list.append([r_score])
            
            ###### sacre blue ######
            sacre_blue_score = sacre_blue_score + bleu.sentence_score(hyp,[ref]).score
        
        cummulative_sacre_bleu_avg_score = sacre_blue_score/float(no_of_news)
        return cummulative_sacre_bleu_avg_score, r_list
    

    def blue_score_calculator(self, model, validation_file_name, no_of_validation_sample, validation_step_size):
        #In validation don't repalce with random words
        number_words_to_replace=0
        temp_gen = self.data_generator(validation_file_name,validation_step_size,number_words_to_replace, model)        
            
        total_sacre_score=0
        blue_batches = 0
        val_batch_loss = []
        rouge_scores_list = []
        blue_number_of_batches = math.floor(no_of_validation_sample / validation_step_size)
        for X_val, y_val in temp_gen:
            score = model.evaluate(X_val, y_val, verbose=0)
            val_batch_loss.append(score)
            y_predicated = np.argmax(model.predict(X_val,batch_size = validation_step_size,verbose = 1), axis=-1)
            y_predicated_words = self.indexes_to_words(y_predicated)
            list_of_word_headline = self.indexes_to_words(self.OHE_to_indexes(y_val))
            assert len(y_val)==len(list_of_word_headline)
                
            cummulative_sacre_bleu, r_list = self.blue_score_text(list_of_word_headline, y_predicated_words)
            total_sacre_score = total_sacre_score + cummulative_sacre_bleu
            rouge_scores_list.append(r_list)
            blue_batches += 1
            if blue_batches >=  blue_number_of_batches:
                #get out of infinite loop of val generator
                break
            if blue_batches%10==0:
                print ("Val - eval for {} out of {}".format(blue_batches, blue_number_of_batches))
         
        del temp_gen
        sacre_bleu_score = total_sacre_score/float(blue_batches)
        return sacre_bleu_score , val_batch_loss , rouge_scores_list    
    
    def rouge_score_calculator(self, rouge_scores_list,no_of_validation_sample,validation_step_size):
        r1_f_score = r2_f_score = rl_f_score = 0.0
        number_of_batches = math.floor(no_of_validation_sample / validation_step_size)
        no_of_samples = number_of_batches*validation_step_size
        for each_batch in rouge_scores_list:
            for each_output in each_batch:
                scores_dic = each_output[0]
                r1_f_score = r1_f_score + scores_dic['rouge-1']['f']
                r2_f_score = r2_f_score + scores_dic['rouge-2']['f']
                rl_f_score = rl_f_score + scores_dic['rouge-l']['f']
        rouge_scores_for_epoch = [r1_f_score/no_of_samples,r2_f_score/no_of_samples , rl_f_score/no_of_samples]
        return rouge_scores_for_epoch
    
    def train(self, model,data_file,val_file,train_size,train_batch_size,val_size,val_step_size,epochs,words_replace_count,model_weights_file_name):
        """
        trains a model
        Manually loop (without using internal epoch parameter of keras),
        train model for each epoch, evaluate logloss and BLUE score of model on validation data
        save model if BLUE/logloss score improvement ...
        save score history for plotting purposes.
        Note : validation step size meaning over here is different from keras
        here validation_step_size means, batch_size in which blue score evaluated
        after all batches processed, blue score s over all batches averaged to get one blue score.
        """
        #load model weights if file present 
        if os.path.isfile(model_weights_file_name):
            print ("loading weights already present in {}".format(model_weights_file_name))
            model.load_weights(model_weights_file_name)
            print ("model weights loaded for further training")
        
        results_list = []
        train_results_list = []
        rougeL_score_track = -1.0
        number_of_batches = math.floor(train_size / float(train_batch_size))
            
        for each_epoch in range(epochs):
            train_loss_list = []
            print ("Running for epoch: ", each_epoch)
            start_time = time.time()        
            train_data = self.data_generator(data_file,train_batch_size,words_replace_count, model)
            batches = 0
            for X_batch, Y_batch in train_data:
                history = model.fit(X_batch,Y_batch,batch_size=train_batch_size,epochs=1)
                batches += 1
                if batches >= number_of_batches :
                    break
                if batches%10==0:
                    print ("training for {} out of {} for epoch {}".format(batches, number_of_batches, each_epoch))
                train_loss_list.append(history.history['loss'][0])
            del train_data
            
            end_time = time.time()
            print("Time to train an epoch: ",end_time-start_time)
            
            train_loss = sum(train_loss_list)/len(train_loss_list)
            train_obj = {
                'epoch_no':each_epoch,
                'train_loss': train_loss
            }
            train_results_list.append(train_obj)
            df1 = pd.DataFrame(data=train_results_list)
            df1.to_csv('./train_results_epoch'+ each_epoch+'.csv')
    
            model.save_weights('./model_weights_' + each_epoch + '.h5')
            print('*********** Model Weights Saved ***************\n')
            
            
            # Evaluating Model on Validation data...
            print('*********** Validation Started **************** \n')
            sacre_bleu_score, val_batch_loss, rouge_scores_list = self.blue_score_calculator(model,val_file,val_size,val_step_size)
            rouge_scores_for_epoch = self.rouge_score_calculator(rouge_scores_list,val_size,val_step_size)
            
            valid_loss = sum(val_batch_loss)/len(val_batch_loss)
    
            if rougeL_score_track < rouge_scores_for_epoch[2]:
                rougeL_score_track = rouge_scores_for_epoch[2]
                print ("best rouge-L score till now ",rougeL_score_track)
            
            
            obj = {
                'epoch_no':each_epoch,
                'train_loss': train_loss,
                'validation_loss': valid_loss,
                'rouge_1': rouge_scores_for_epoch[0],
                'rouge_2': rouge_scores_for_epoch[1],
                'rouge_l': rouge_scores_for_epoch[2],
                'sacre_bleu_score': sacre_bleu_score
            }
            results_list.append(obj)
            df2 = pd.DataFrame(data=results_list)
            df2.to_csv('./final_results_epoch_'+each_epoch+'.csv')
    
            print("epoch : {0}, train : {1}, val : {2}, rouge_l : {3}".format(each_epoch, train_loss, valid_loss, rouge_scores_for_epoch[2]))
    
    ## Inference Modules
    def is_headline_end(self, word_index_list,current_predication_position):
        """
        is headline ended checker
        current_predication_position is 0 index based
        """
        if (word_index_list is None) or (len(word_index_list)==0):
            return False
        if word_index_list[current_predication_position]==eos_tag_location or current_predication_position>=max_length:
            return True
        return False
    
    def process_word(self, predication, word_position_index, top_k, X, prev_layer_log_prob, alpha_value):
        """
        Extract top k predications of given position
        """
        #predication contains only one element
        #shape of predication (1,max_head_line_words,vocab_size)
        predication = predication[0]
        #predication shape = (max_head_line_words,vocab_size)
        predication_at_word_index = predication[word_position_index]
        # predication_at_word_index shape = (vocab_size,)
        #http://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
        sorted_arg = predication_at_word_index.argsort()
        top_probable_indexes = sorted_arg[::-1]
        top_probabilities = np.take(predication_at_word_index,top_probable_indexes)
        log_probabilities = np.log(top_probabilities)
        #make sure elements doesnot contain -infinity
        log_probabilities[log_probabilities == -inf] = -sys.maxsize - 1
        #add prev layer probability
        log_probabilities = log_probabilities + prev_layer_log_prob
        log_probabilities = log_probabilities/pow((word_position_index+1), alpha_value)
        log_probabilities = np.sort(log_probabilities)[::-1]
        assert len(log_probabilities)==len(top_probable_indexes)
            
        #add previous words ... preparation for next input
        #offset calculate ... description + eos + headline till now
        offset = max_len_desc+word_position_index+1
        ans = []
        count = 0 
        for i,j in zip(log_probabilities, top_probable_indexes):
            #check for word should not repeat in headline ... 
            #checking for last x words, where x = dont_repeat_word_in_last
            if j in X[max_len_desc+1:offset][-dont_repeat_word_in_last:]:
                continue
            #if (word_position_index < min_head_line_gen) and (j in [empty_tag_location, unknown_tag_location, eos_tag_location]):
            #    continue
            next_input = np.concatenate((X[:offset], [j,]))
            next_input = next_input.reshape((1,next_input.shape[0]))
            #for the last time last word put at max_length + 1 position 
            #don't truncate that
            if offset!=max_length:
                next_input = sequence.pad_sequences(next_input, maxlen=max_length, value=empty_tag_location, padding='post', truncating='post')
            next_input = next_input[0]
            count = count + 1
            ans.append((i,next_input))
            #print("answer list",ans)
            if count>=top_k:
                break
        #[(prob,list_of_words_as_next_input),(prob2,list_of_words_as_next_input2),...]
        return ans
    
    def beam_search(self, model, X, top_k, alpha_value):
        """
        1.Loop over max headline word allowed
        2.predict word prob and select top k words for each position
        3.select top probable combination uptil now for next round
        """
        #contains [(log_p untill now, word_seq), (log_p2, word_seq2)]
        prev_word_index_top_k = []
        curr_word_index_top_k = []
        done_with_pred = []
        #1d => 2d array [1,2,3] => [[1,2,3]]
        data = X.reshape((1,X.shape[0]))
        #shape of predication (1,max_head_line_words,vocab_size)
        predication = model.predict(data,verbose=0)
        #prev layer probability 1 => np.log(0)=0.0
        prev_word_index_top_k = self.process_word(predication,0,top_k,X,0.0, alpha_value)
            
        #1st time its done above to fill prev word therefore started from 1
        for i in range(1,max_len_head):
            #i = represents current intrested layer ...
            for j in range(len(prev_word_index_top_k)):
                #j = each time loops for top k results ...
                probability_now, current_intput = prev_word_index_top_k[j]
                data = current_intput.reshape((1,current_intput.shape[0]))
                predication = model.predict(data,verbose=0)
                next_top_k_for_curr_word = self.process_word(predication,i,top_k,current_intput,probability_now, alpha_value)
                curr_word_index_top_k = curr_word_index_top_k + next_top_k_for_curr_word
                    
            #sort new list, empty old, copy top k element to old, empty new
            curr_word_index_top_k = sorted(curr_word_index_top_k,key=itemgetter(0),reverse=True)
            prev_word_index_top_k_temp = curr_word_index_top_k[:top_k]
            curr_word_index_top_k = []
            prev_word_index_top_k = []
            #if word predication eos ... put it done list ...
            for each_proba, each_word_idx_list in prev_word_index_top_k_temp:
                offset = max_len_desc+i+1
                if self.is_headline_end(each_word_idx_list,offset):
                    done_with_pred.append((each_proba, each_word_idx_list))
                else:
                    prev_word_index_top_k.append((each_proba,each_word_idx_list))
                
        #sort according to most probable
        done_with_pred = sorted(done_with_pred,key=itemgetter(0),reverse=True)
        done_with_pred = done_with_pred[:top_k]
        return done_with_pred
    
    
    def test(self, model, data_file_name, no_of_testing_sample,test_batch_size, model_weights_file_name,top_k,output_file, alpha_value, seperator='#|#'):
        """
        test on given description data file with empty headline ...
        """
        model.load_weights(model_weights_file_name)
        print ("model weights loaded")
        #test_batch_size = 1
        test_data_generator = self.data_generator(data_file_name,test_batch_size,number_words_to_replace=0, model=None,is_training=False)
        number_of_batches = math.ceil(no_of_testing_sample / float(test_batch_size))
        d = {}
        actual_headlines_list = []
        predicted_headlines_list = []
        with codecs.open(output_file, 'w',encoding='utf8') as f:
            #testing batches
            batches = 0
            for X_batch, Y_batch in test_data_generator:
                #Always come one because X_batch contains one element
                X = X_batch[0]
                Y = Y_batch[0]
                assert X[max_len_desc]==eos_tag_location
                #wipe up news headlines present and replace by empty tag ...            
                X[max_len_desc+1:]=empty_tag_location
                result = self.beam_search(model,X,top_k, alpha_value)
                #take top most probable element
                list_of_word_indexes = result[0][1]
                #print("list_of_word_indexes",list_of_word_indexes)
                list_of_word_indexes = list_of_word_indexes[max_len_desc+1:]
                list_of_words = self.indexes_to_words([list_of_word_indexes])[0]
                #print("list_of_words",list_of_words)
                if self.use_bpe:
                    headline = self.emb_model.decode(list_of_words)
                else:
                    headline = u" ".join(list_of_words)
                print(Y+seperator+headline+"\n")
                f.write(Y+seperator+headline+"\n")
                actual_headlines_list.append(Y)
                predicted_headlines_list.append(headline)
                batches += 1
                #take last chunk and roll over to start ...
                #therefore float used ... 
                if batches >= number_of_batches :
                    break
                if batches%10==0:
                    print ("working on batch no {} out of {}".format(batches,number_of_batches))
            d["Actual_headline"] = actual_headlines_list
            d["Predicted_headline"] = predicted_headlines_list
            df = pd.DataFrame(d)
            df.to_csv("./test_predictions.csv", index=False)

def get_file_size(file):
    line_count = 0
    with open(file, 'r', encoding='utf-8') as fp:
        for each_line in fp:
            line_count+=1
    return line_count
    

    

if __name__ == '__main__':
    
    # Command line arguments.
    parser = argparse.ArgumentParser(description="Input arguments for HG model.")
    parser.add_argument('--vocab_csv', required=True, help='vocabulary csv file path' )
    parser.add_argument('--train_file', required=True, help='train data file in .jsonl format')
    parser.add_argument('--dev_file', required=True, help='dev data file in .jsonl format')
    parser.add_argument('--test_file', help='test data file in .jsonl format')
    parser.add_argument('--model_weights', required=True, help='model weights file path(model_weights.h5)')
    parser.add_argument('--test_outputs_file', default='./test_predictions.txt', help='test predictions txt file path')
    parser.add_argument('--fastText_embeddings_file' ,help="fastText embeddings file path, ex. ./cc.te.300.bin")
    parser.add_argument('--use_bpe', required=True, default='False', choices=['True','False'])
    parser.add_argument('--bpe_lang_code', choices=['te','ta','kn','ml','hi','bn','mr','gu'], help='language code for bpe model')
    parser.add_argument('--bpe_vocab_size', type=int, default=50000, choices=[1000,3000,5000,10000,25000,50000,100000,200000], help='Specify the bpe vocabulary size')
    parser.add_argument('--language', choices=['telugu','tamil','kannada','malayalam','hindi','bengali','marathi','gujarati'], help='language name')
    parser.add_argument('--do_train', required=True, choices=['True','False'])
    parser.add_argument('--do_test', required=True, choices=['True','False'])
    parser.add_argument('--train_batch_size', type=int, default=8, help='Specify the train batch size')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Specify the validation batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Specify the test batch size')
    parser.add_argument('--epochs', type=int, default=1, help="Specify the number of epochs")
    parser.add_argument('--rnn_type', type=str, required=True, default="gru", choices=['gru','lstm'], help='Specify the RNN Type')
    parser.add_argument('--max_input_length', type=int, required=True, default=300, help="Specify the maximum length of input text sequence")
    parser.add_argument('--max_target_length', type=int, required=True, default=30, help='Specify the maximum target sequence length')
    parser.add_argument('--beam_size', type=int, default=5, help='Specify the beam width in beam search')
    parser.add_argument('--beamsearch_length_penalty', type=float, default=0.0, help="Specify the length penalty(alpha_value) for beam search")
    args = parser.parse_args()

    vocabulary_file_path = args.vocab_csv
    train_file_path = args.train_file
    dev_file_path = args.dev_file
    test_file_path = args.test_file
    model_weights_path = args.model_weights
    test_outputs_path = args.test_outputs_file
    fastText_embeddings_file = args.fastText_embeddings_file
    use_bpe = args.use_bpe
    bpe_lang_code = args.bpe_lang_code
    bpe_vocab_size = args.bpe_vocab_size
    language = args.language
    do_train = args.do_train
    do_test = args.do_test 
    train_batch_size = args.train_batch_size
    val_step_size = args.val_batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    rnn_type = args.rnn_type
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length
    beam_size = args.beam_size
    alpha_value = args.beamsearch_length_penalty

    
    train_size = get_file_size(train_file_path)
    val_size = get_file_size(dev_file_path)
    no_of_testing_sample = get_file_size(test_file_path)
    #test_batch_size = 1
    
    
    # Loading the vocabulary file.
    vocab_df = pd.read_csv(vocabulary_file_path)
    print("Vocabulary Size : ", vocab_df.shape[0])
    
    if use_bpe=='True':
        # Loading BPE embeddings
        emb_model = BPEmb(lang=bpe_lang_code, vs=bpe_vocab_size, dim=300)
    else:
        # Loading FastText embeddings.
        emb_model = fasttext.load_model(fastText_embeddings_file)
    
    ## Multilingual rouge score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, lang=language)
    
    # Model hyperparameters.
    # -------------------------------------------------------------------------
    seed = 28
    random.seed(seed)
    np.random.seed(seed)

    top_freq_word_to_use = 40000
    embedding_dimension = 300
    max_len_head = max_target_length
    max_len_desc = max_input_length
    max_length = max_len_head + max_len_desc
    min_head_line_gen = 8
    dont_repeat_word_in_last = 5
    words_replace_count = 5
    
    rnn_layers = 4
    rnn_size = 600
    activation_rnn_size = 50
    learning_rate = 1e-4
    beam_size = beam_size
    # alpha value specifies the length penalty for beam search.
    alpha_value = alpha_value
    
    empty_tag_location = 0
    eos_tag_location = 1
    unknown_tag_location = 2
    # -------------------------------------------------------------------------
    
    
    hg = HeadlineGeneration(vocab_df=vocab_df, use_bpe=False, emb_model=emb_model, scorer=scorer, rnn_type=rnn_type)
    hg.read_word_embedding()
    model = hg.create_model()
    
    if do_train=="True":
        print("****** Training Started ********* \n")
        print("Train size : {0}, Train batch size : {1}".format(train_size, train_batch_size))
        print("Validation size : {0}, Val batch size : {1}".format(val_size, val_step_size))
        print("No.of epochs : {0}\n".format(epochs))
        hg.train(model=model, 
                data_file = train_file_path, 
                val_file = dev_file_path, 
                train_size = train_size, 
                train_batch_size = train_batch_size,
                val_size = val_size,
                val_step_size = val_step_size, 
                epochs = epochs, 
                words_replace_count = words_replace_count,
                model_weights_file_name = model_weights_path)
    if do_test=="True":
        print("****** Testing Started ********* \n")
        print("Test size : {0}, Test batch size {1}".format(no_of_testing_sample, test_batch_size))
        hg.test(model = model,
               data_file_name = test_file_path,
               no_of_testing_sample = no_of_testing_sample,
               test_batch_size=test_batch_size,
               model_weights_file_name = model_weights_path,
               top_k = beam_size,
               output_file = test_outputs_path,
               alpha_value=alpha_value)
    
    if do_train=="False":
        if do_test=="False":
            print("You must set TRUE for either do_train or do_test argument. You can find more details about the arguments by passing '--help' as an argument.")        
    