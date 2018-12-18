"""
Attention mechanisms
Based on paper "Attention Is All You Need"
https://arxiv.org/abs/1706.03762

"""
import numpy as np

class DotAttn:
    """
    key - value mode
    """
    def __init__(self):
        self.enco_key = None
        self.enco_value = None
        self.deco_key = None
        self.deco_filters = None
        self.deco_timestep = None
        self.batch = None
        self.depth = None
        self.enco_filters = None
        self.enco_timestep = None
        self.score = None
        self.alpha = None
    def forward(self, enco_key, enco_value, deco_key):
        """
        input encoder key, encoder value, decoder key data
        """
        self.enco_key = enco_key
        self.enco_value = enco_value
        self.deco_key = deco_key
        self.deco_filters, self.deco_timestep, self.batch, self.depth = deco_key.shape
        self.enco_filters, self.enco_timestep = enco_key.shape[:2]

        score = np.zeros((self.deco_filters,
                          self.deco_timestep,
                          self.batch,
                          self.enco_timestep)).astype(np.float32)
        output = np.zeros_like(self.enco_value)

        # calculate dot
        for f in range(self.deco_filters):
            for b in range(self.batch):
                score[f, :, b, :] = np.dot(self.deco_key[f, :, b, :],
                                           self.enco_key[f, :, b, :].T)
        max_score_tile = np.tile(np.max(score, axis=3),
                                 (self.enco_timestep, 1, 1, 1)).transpose(1, 2, 3, 0)

        self.score = np.exp(score- max_score_tile)
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

    def forward(self, inp):
        if not self.forward_pass_once:
            self.attn_enco_value = inp[:, :, :, :self.cutoff_at]
            self.attn_enco_key = inp[:, :, :, self.cutoff_at:]
            self.forward_pass_once = True
            return  self.attn_enco_value
        if  self.forward_pass_once:
            attn_deco_value = inp[:, :, :, :self.cutoff_at]
            attn_deco_key = inp[:, :, :, self.cutoff_at:]
            output_attn = self.attention_model.forward(self.attn_enco_key,
                                                       self.attn_enco_value,
                                                       attn_deco_key)
            self.forward_pass_once = False
            return  np.concatenate((attn_deco_value, output_attn), axis=3)
    def backprop(self, dLoss):
        if not self.backprop_pass_once:
            dLoss_deco_value = dLoss[:, :, :, :self.cutoff_at]
            dLoss_attn = dLoss[:, :, :, self.cutoff_at:]
            self.dLoss_enco_key, self.dLoss_enco_value, dLoss_deco_key = \
                self.attention_model.backprop(dLoss_attn)
            self.backprop_pass_once = True
            return  np.concatenate((dLoss_deco_value, dLoss_deco_key), axis=3)
        if  self.backprop_pass_once:
            self.backprop_pass_once = False
            return np.concatenate((dLoss+self.dLoss_enco_value, self.dLoss_enco_key), axis=3)
