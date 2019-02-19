import numpy as np

class DotAttn:
    """
    key - value mode
    """
    def __init__(self):
        pass
    def forward(self, enco_key, enco_value, deco_key):
        """
        input encoder/decoder key data
        """
        self.enco_key = enco_key
        self.enco_value = enco_value
        self.deco_key = deco_key
        self.deco_filters, self.deco_timestep, self.batch, self.depth = deco_key.shape
        self.enco_filters, self.enco_timestep = enco_key.shape[:2]

        score = np.zeros((self.deco_filters, self.deco_timestep,
                          self.batch, self.enco_timestep)).astype(np.float32)
        output = np.zeros_like(self.enco_value)

        # calculate dot
        for f in range(self.deco_filters):
            for b in range(self.batch):
                score[f, :, b, :] = np.dot(self.deco_key[f, :, b, :], self.enco_key[f, :, b, :].T)
        max_score_tile = np.tile(np.max(score, axis=3),
                                 (self.enco_timestep, 1, 1, 1)).transpose(1, 2, 3, 0)

        self.score = np.exp(score - max_score_tile)
        sum_exp = np.sum(self.score, axis=3)

        sum_exp_tile = np.tile(sum_exp, (self.enco_timestep, 1, 1, 1)).transpose(1, 2, 3, 0)
        self.alpha = self.score/sum_exp_tile
        for f in range(self.deco_filters):
            for b in range(self.batch):
                output[f, :, b, :] = np.dot(self.alpha[f, :, b, :], self.enco_value[f, :, b, :])
        return output
    def backprop(self, dLoss):
        "upper case: decoder, lower case : encoder/both"
        dLoss_enco_value = np.einsum('fTbt,fTbd->ftbd', self.alpha, dLoss)
        dLoss_deco_key = np.zeros_like(self.deco_key)
        dLoss_enco_key = np.zeros_like(self.enco_key)
        for f in range(self.deco_filters):
            this_dLoss = dLoss[f]
            for h1 in range(self.enco_timestep):
                this_alpha = self.alpha[f, :, :, h1]
                this_dLoss_enco_value = np.einsum('Dbd,bd->Db',
                                                  this_dLoss,
                                                  self.enco_value[f, h1])
                dLoss_deco_key[f] += np.einsum('Db,bd->Dbd',
                                               this_alpha*this_dLoss_enco_value,
                                               self.enco_key[f, h1])
                dLoss_enco_key[f, h1] += np.einsum('Db,Dbd->bd',
                                                   this_alpha*this_dLoss_enco_value,
                                                   self.deco_key[f])
                for h2 in range(self.enco_timestep):
                    corr_alpha = this_alpha*self.alpha[f, :, :, h2]
                    dLoss_deco_key[f] -= np.einsum('Db,bd->Dbd',
                                                   corr_alpha*this_dLoss_enco_value,
                                                   self.enco_key[f, h2])
                    dLoss_enco_key[f, h2] -= np.einsum('Db,Dbd->bd',
                                                       corr_alpha*this_dLoss_enco_value,
                                                       self.deco_key[f])
        return  dLoss_enco_key, dLoss_enco_value, dLoss_deco_key

    def get_alpha(self):
        return self.alpha



class Attn_helper:
    """
    pass the model twice, the first time attn_branch
    1st time  input -> value1, key1. Remember both, while only pass value.
    2nd time  input -> value2, key2. Remember both, pass np.concat(value , attn(key1,value1,key2))
    """
    def __init__(self, attention_model, cutoff_at):
        self.attention_model = attention_model
        self.cutoff_at = cutoff_at

        self.attn_enco_value = None
        self.attn_enco_key = None

        self.dLoss_enco_value = None
        self.dLoss_enco_key = None

        self.forward_pass_once = False
        self.backprop_pass_once = False

        self.attn_enco_value_length = 0

    def forward(self, inp):
        if not self.forward_pass_once:
            self.attn_enco_value_length = inp.shape[3] - self.cutoff_at
            self.attn_enco_value = inp[:, :, :, :-self.cutoff_at]
            self.attn_enco_key = inp[:, :, :, -self.cutoff_at:]
            self.forward_pass_once = True
            return  self.attn_enco_value
        if  self.forward_pass_once:
            attn_deco_value = inp[:, :, :, :-self.cutoff_at]
            attn_deco_key = inp[:, :, :, -self.cutoff_at:]
            output_attn = self.attention_model.forward(self.attn_enco_key,
                                                       self.attn_enco_value,
                                                       attn_deco_key)
            self.forward_pass_once = False
            return  np.concatenate((attn_deco_value, output_attn), axis=3)
    def backprop(self, dLoss):
        if not self.backprop_pass_once:
            dLoss_deco_value = dLoss[:, :, :, :-self.attn_enco_value_length]
            dLoss_attn = dLoss[:, :, :, -self.attn_enco_value_length:]
            self.dLoss_enco_key, self.dLoss_enco_value, dLoss_deco_key = \
                self.attention_model.backprop(dLoss_attn)
            self.backprop_pass_once = True
            return  np.concatenate((dLoss_deco_value, dLoss_deco_key), axis=3)
        if  self.backprop_pass_once:
            self.backprop_pass_once = False
            return np.concatenate((dLoss+self.dLoss_enco_value, self.dLoss_enco_key), axis=3)

class DotAttn_3d:
    """
    key - value mode
    """
    def __init__(self):
        self._alpha = []
        self._deco_key = []
        self.use_timestep = False
    def forward(self, enco_key, enco_value, deco_key):
        """
        input encoder/decoder key data
        """
        self.enco_key = enco_key
        self.enco_value = enco_value
        self.deco_key = deco_key
        self.deco_timestep, self.batch, self.depth = deco_key.shape
        self.enco_timestep = enco_key.shape[0]
        self.value_depth = self.enco_value.shape[2]

        score = np.zeros((self.deco_timestep, self.batch, self.enco_timestep)).astype(np.float32)
        output = np.zeros((self.deco_timestep, self.batch, self.value_depth)).astype(np.float32)
        # calculate dot
        for b in range(self.batch):
            score[:, b, :] = np.dot(self.deco_key[:, b, :], self.enco_key[:, b, :].T)
        max_score_tile = np.tile(np.max(score, axis=2),
                                 (self.enco_timestep, 1, 1)).transpose(1, 2, 0)
        self.score = np.exp(score - max_score_tile)
        sum_exp = np.sum(self.score, axis=2)

        sum_exp_tile = np.tile(sum_exp, (self.enco_timestep, 1, 1)).transpose(1, 2, 0)
        self.alpha = self.score/sum_exp_tile
        for b in range(self.batch):
            output[:, b, :] = np.dot(self.alpha[:, b, :], self.enco_value[:, b, :])
        return output
    def timestep_forward(self, enco_key, enco_value, deco_key):
        """
        input encoder/decoder key data
        """
        self.use_timestep = True
        self.enco_key = enco_key
        self.enco_value = enco_value
        self._deco_key.append(deco_key)
        self.batch, self.depth = deco_key.shape
        self.enco_timestep = enco_key.shape[0]

        score = np.zeros((self.batch, self.enco_timestep)).astype(np.float32)
        output = np.zeros_like(self.enco_value[0])

        # calculate dot
        for b in range(self.batch):
            score[b, :] = np.dot(deco_key[b, :], self.enco_key[:, b, :].T)
        max_score_tile = np.tile(np.max(score, axis=1), (self.enco_timestep, 1)).transpose(1, 0)

        self.score = np.exp(score- max_score_tile)
        sum_exp = np.sum(self.score, axis=1)

        sum_exp_tile = np.tile(sum_exp, (self.enco_timestep, 1)).transpose(1, 0)
        this_alpha = self.score/sum_exp_tile
        self._alpha.append(this_alpha)
        for b in range(self.batch):
            output[b, :] = np.dot(this_alpha[b, :], self.enco_value[:, b, :])
        return output

    def timestep_gather(self):
        self.deco_key = np.array(self._deco_key)
        self.alpha = np.array(self._alpha)
        self._deco_key = []
        self._alpha = []
    def backprop(self, dLoss):
        "upper case: decoder, lower case : encoder/both"
        if self.use_timestep:
            self.timestep_gather()
        dLoss_enco_value = np.einsum('Tbt,Tbd->tbd', self.alpha, dLoss)
        dLoss_deco_key = np.zeros_like(self.deco_key)
        dLoss_enco_key = np.zeros_like(self.enco_key)
        this_dLoss = dLoss
        for h1 in range(self.enco_timestep):
            this_alpha = self.alpha[:, :, h1]
            this_dLoss_enco_value = np.einsum('Dbd,bd->Db',
                                              this_dLoss,
                                              self.enco_value[h1])
            dLoss_deco_key += np.einsum('Db,bd->Dbd',
                                        this_alpha*this_dLoss_enco_value,
                                        self.enco_key[h1])
            dLoss_enco_key[h1] += np.einsum('Db,Dbd->bd',
                                            this_alpha*this_dLoss_enco_value,
                                            self.deco_key)
            for h2 in range(self.enco_timestep):
                corr_alpha = this_alpha*self.alpha[:, :, h2]
                dLoss_deco_key -= np.einsum('Db,bd->Dbd',
                                            corr_alpha*this_dLoss_enco_value,
                                            self.enco_key[h2])
                dLoss_enco_key[h2] -= np.einsum('Db,Dbd->bd',
                                                corr_alpha*this_dLoss_enco_value,
                                                self.deco_key)
        return  dLoss_enco_key, dLoss_enco_value, dLoss_deco_key

    def get_alpha(self):
        return self.alpha if not self.use_timestep else np.array(self._alpha)



class LSTM_Attn_helper:
    """
    pass the model twice, the first time attn_branch
    1st time  input -> value1, key1. Remember both, while only pass value.
    2nd time  input -> value2, key2. Remember both, pass np.concat(value , attn(key1,value1,key2))
    """
    def __init__(self, attention_model, cutoff_at):
        self.attention_model = attention_model
        self.cutoff_at = cutoff_at

        self.attn_enco_value = None
        self.attn_enco_key = None

        self.dLoss_enco_value = None
        self.dLoss_enco_key = None

        self.pass_encoder = False
        self.backprop_pass_once = False

        self.attn_enco_value_length = 0

    def forward(self, inp):
        if not self.pass_encoder:
            self.attn_enco_value_length = inp.shape[2] - self.cutoff_at
            self.attn_enco_value = inp[:, :, :-self.cutoff_at]
            self.attn_enco_key = inp[:, :, -self.cutoff_at:]
            self.pass_encoder = True
            return  self.attn_enco_value
        if  self.pass_encoder:
            attn_deco_value = inp[:, :, :-self.cutoff_at]
            attn_deco_key = inp[:, :, -self.cutoff_at:]
            output_attn = self.attention_model.forward(self.attn_enco_key,
                                                       self.attn_enco_value,
                                                       attn_deco_key)
            return  np.concatenate((attn_deco_value, output_attn), axis=2)
    def timestep_forward(self, inp):
        if not self.pass_encoder:
            self.attn_enco_value_length = inp.shape[2] - self.cutoff_at
            self.attn_enco_value = inp[:, :, :-self.cutoff_at]
            self.attn_enco_key = inp[:, :, -self.cutoff_at:]
            self.pass_encoder = True
            return  self.attn_enco_value
        "decoder"
        if  self.pass_encoder:
            attn_deco_value = inp[:, :-self.cutoff_at]
            attn_deco_key = inp[:, -self.cutoff_at:]
            output_attn = self.attention_model.timestep_forward(self.attn_enco_key,
                                                                self.attn_enco_value,
                                                                attn_deco_key)
            return np.concatenate((attn_deco_value, output_attn), axis=1)
    def timestep_gather(self):
        self.pass_encoder = False
        self.attention_model.timestep_gather()
    def get_alpha(self):
        return self.attention_model.get_alpha()
    def backprop(self, dLoss):
        self.pass_encoder = False
        if not self.backprop_pass_once:
            dLoss_deco_value = dLoss[:, :, :-self.attn_enco_value_length]
            dLoss_attn = dLoss[:, :, -self.attn_enco_value_length:]
            self.dLoss_enco_key, self.dLoss_enco_value, dLoss_deco_key = \
                self.attention_model.backprop(dLoss_attn)
            self.backprop_pass_once = True
            return  np.concatenate((dLoss_deco_value, dLoss_deco_key), axis=2)
        if  self.backprop_pass_once:
            self.backprop_pass_once = False
            return np.concatenate((self.dLoss_enco_value, self.dLoss_enco_key), axis=2)

class LSTM_MultiAttn:
    """
    pass the model twice, the first time attn_branch
    1st time  input ->(en_value1, en_value2 ... en_key1 en_key2 ...) 
    2nd time  input ->(de_values, de_key1, de_key2, ...) 
                      return np.concat(de_values , attn1(en_key1,en_value1,de_key1), attn2(...) )
    """
    def __init__(self, heads, value_depth, key_depth, dtype=np.float32):
        """
        heads: how many attention heads.
        value_depth: 
        key_depth: 
        """
        self.heads = heads
        
        self.attention_models = []
        for _ in range(self.heads):
            self.attention_models.append(DotAttn_3d())

        self.attn_enco_value = None
        self.attn_enco_key = None
        self.attn_deco_value = None
        self.attn_deco_key = None

        self.dLoss_enco_value = None
        self.dLoss_enco_key = None

        self.pass_encoder = False
        self.backprop_pass_once = False

        self.cutoff_at = self.heads*key_depth
        self.attn_enco_value_length = self.heads*value_depth
        self.value_depth = value_depth
        self.key_depth = key_depth
        self.dtype = dtype
    def forward(self, inp):
        if not self.pass_encoder:
            self.attn_enco_value = inp[:, :, :-self.cutoff_at]
            self.attn_enco_key = inp[:, :, -self.cutoff_at:]
            self.pass_encoder = True
            return  self.attn_enco_value
        if  self.pass_encoder:
            self.attn_deco_value = inp[:, :, :-self.cutoff_at]
            self.attn_deco_key = inp[:, :, -self.cutoff_at:]
            output_attn = np.zeros((self.attn_deco_value.shape[0], self.attn_deco_value.shape[1],
                                    self.attn_enco_value.shape[2])).astype(self.dtype)
            for this_head in range(self.heads):
                key_start = this_head*self.key_depth
                key_end = (1+this_head)*self.key_depth
                value_start = this_head*self.value_depth
                value_end = (1+this_head)*self.value_depth
                output_attn[:, :, value_start:value_end] = \
                    self.attention_models[this_head].forward(self.attn_enco_key[:, :, key_start:key_end],                
                                                             self.attn_enco_value[:, :, value_start:value_end],
                                                             self.attn_deco_key[:, :, key_start:key_end])
            return  np.concatenate((self.attn_deco_value, output_attn), axis=2)
    def timestep_forward(self, inp):
        if not self.pass_encoder:
            self.attn_enco_value = inp[:, :, :-self.cutoff_at]
            self.attn_enco_key = inp[:, :, -self.cutoff_at:]
            self.pass_encoder = True
            return  self.attn_enco_value
        if  self.pass_encoder:
            attn_deco_value = inp[:, :-self.cutoff_at]
            attn_deco_key = inp[:, -self.cutoff_at:]
            output_attn = np.zeros_like(self.attn_enco_value[0])
            for this_head in range(self.heads):
                key_start = this_head*self.key_depth
                key_end = (1+this_head)*self.key_depth
                value_start = this_head*self.value_depth
                value_end = (1+this_head)*self.value_depth
                output_attn[:, value_start:value_end] = \
                    self.attention_models[this_head].timestep_forward(self.attn_enco_key[:, :, key_start:key_end],
                                                                      self.attn_enco_value[:, :, value_start:value_end],
                                                                      attn_deco_key[:, key_start:key_end])

            return np.concatenate((attn_deco_value, output_attn), axis=1)

    def timestep_gather(self):
        self.pass_encoder = False
        for this_head in range(self.heads):
            self.attention_models[this_head].timestep_gather()
    def get_alpha(self):
        alphas = []
        for this_head in range(self.heads):
            alphas.append(self.attention_models[this_head].get_alpha())
        return alphas
    def backprop(self, dLoss):
        self.pass_encoder = False
        if not self.backprop_pass_once:
            dLoss_deco_value = dLoss[:, :, :-self.attn_enco_value_length]
            dLoss_attn = dLoss[:, :, -self.attn_enco_value_length:]

            self.dLoss_enco_key = np.zeros_like(self.attn_enco_key)
            self.dLoss_enco_value = np.zeros_like(self.attn_enco_value)
            dLoss_deco_key = np.zeros((dLoss.shape[0],
                                       dLoss.shape[1], self.heads*self.key_depth)).astype(self.dtype)
            for this_head in range(self.heads):
                key_start = this_head*self.key_depth
                key_end = (1+this_head)*self.key_depth
                value_start = this_head*self.value_depth
                value_end = (1+this_head)*self.value_depth
                _dLoss_enco_key, _dLoss_enco_value, _dLoss_deco_key = \
                    self.attention_models[this_head].backprop(dLoss_attn[:, :, value_start:value_end])
                self.dLoss_enco_key[:, :, key_start:key_end] = _dLoss_enco_key
                self.dLoss_enco_value[:, :, value_start:value_end] = _dLoss_enco_value
                dLoss_deco_key[:, :, key_start:key_end] = _dLoss_deco_key
            self.backprop_pass_once = True
            return  np.concatenate((dLoss_deco_value, dLoss_deco_key), axis=2)
        if  self.backprop_pass_once:
            self.backprop_pass_once = False
            return np.concatenate((self.dLoss_enco_value, self.dLoss_enco_key), axis=2)


class BiLSTM_dotAttn_wrapper():
    """
    input original biLSTM output : (encoder_value, encoder_key, decoder_value, decoder_key)
    BiLSTM_dotAttn_helper return : (encoder_value, decoder_value, encoder_key, decoder_key)
    """
    def __init__(self, value_depth, key_depth):
        self.fw_value_end = value_depth
        self.fw_key_end = value_depth+key_depth
        self.bw_value_end = 2*value_depth+key_depth
        self.value_depth = value_depth
        self.key_depth = key_depth
    def forward(self, en_h_output):
        en_h_output_reranged = np.zeros_like(en_h_output)
        en_h_output_reranged[:, :, :self.value_depth] = en_h_output[:, :, :self.fw_value_end]
        en_h_output_reranged[:, :, self.value_depth:2*self.value_depth] =\
            en_h_output[:, :, self.fw_key_end:self.bw_value_end]
        en_h_output_reranged[:, :, 2*self.value_depth:2*self.value_depth+self.key_depth] =\
            en_h_output[:, :, self.fw_value_end:self.fw_key_end]
        en_h_output_reranged[:, :, 2*self.value_depth+self.key_depth:] =\
            en_h_output[:, :, self.bw_value_end:]
        return en_h_output_reranged
    def backprop(self, dLoss):
        dL_reranged = np.zeros_like(dLoss)
        dL_reranged[:, :, :self.fw_value_end] = dLoss[:, :, :self.value_depth]
        dL_reranged[:, :, self.fw_value_end:self.fw_key_end] =\
            dLoss[:, :, 2*self.value_depth:2*self.value_depth+self.key_depth]
        dL_reranged[:, :, self.fw_key_end:self.bw_value_end] =\
            dLoss[:, :, self.value_depth:2*self.value_depth]
        dL_reranged[:, :, self.bw_value_end:] = dLoss[:, :, 2*self.value_depth+self.key_depth:]
        return dL_reranged
