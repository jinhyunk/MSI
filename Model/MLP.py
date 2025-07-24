from Model.layer import *

class Model(nn.Module):
    def __init__(self,in_l:int=2, out_l:int=3, emb_s:int=256, d:int=8):
        super().__init__()
        self.NN_init = TanhLayer(in_l,emb_s)
        self.NN = TanhLayer(emb_s,emb_s)
        self.Layerlist = []
        
        for i in range (d):
            layer_list = self.NN
            self.Layerlist.append(layer_list)
        self.Layer_list = nn.ModuleList(self.Layerlist)
        self.NN_out = nn.Linear(emb_s,out_l)
        self.depth = d
        
    def forward(self,inputs):
        out = self.NN_init(inputs)
        for layer in self.Layer_list:
            out = layer(out)
        out = self.NN_out(out)
        
        return out 