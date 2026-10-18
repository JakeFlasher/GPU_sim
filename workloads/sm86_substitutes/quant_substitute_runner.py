#!/usr/bin/env python3
"""Row-specific reduced SM86 controls. No native FP8/FP4 MMA is claimed.

All operands are generated once on CPU before the profiler region. Each row
preserves its named operation, with explicit reduced geometry. E4M3 and E2M1
rounding/scales are software operations; GEMMs accumulate FP32 on Ampere.
These are comparable operator controls, not bit-identical Blackwell kernels.
"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

# Explicit mapping avoids accidental matches such as tma inside softmax.
OPS = {
1:'output_bias',2:'qkv_norm',3:'gate_up',4:'weighted_expert',5:'router_projection',
6:'output_bias',7:'qkv_split',8:'multimodal_merge',9:'temporal_merge',10:'output',
11:'group_router',12:'shared_mlp',13:'kv_compression',14:'yarn',15:'output',
16:'mla_qkv',17:'shared_mlp',18:'output_residual',19:'gqa_projected',20:'linear',
21:'mamba_in_projection',22:'mamba_out_projection',23:'ssm_discretize',
24:'output',25:'qkv_split',26:'mamba_out_projection',27:'router_projection',
28:'qkv_rope',29:'shared_mlp',30:'prenorm_attention',31:'gqa_quant_attention',
32:'expert_mlp',33:'softmax_router'}
FP4_IDS={8,9,18,19,20,24,25,26,27,28,29,30,31,32,33}
H=64; I=128; S=8; HEADS=4; KVHEADS=2; D=16

def make_inputs(seed):
    g=torch.Generator().manual_seed(seed)
    def rand(*s):return torch.randn(*s,generator=g,dtype=torch.float32)*.2
    def weight(n,k):return rand(n,k)/math.sqrt(k)
    # Reused shapes are documented; distinct weights/operands still represent
    # the projections independently (in particular Q != K != V).
    return dict(x=rand(S,H),residual=rand(S,H), bias=rand(H),
        wo=weight(H,H),wq=weight(H,H),wk=weight(KVHEADS*D,H),wv=weight(KVHEADS*D,H),
        gate=weight(I,H),up=weight(I,H),down=weight(H,I),
        router=weight(16,H),route_bias=rand(16),routing_weight=torch.rand(S,1,generator=g),
        q_a=weight(16,H),q_b=weight(H,16),kv_a=weight(32,H),kv_b=weight(HEADS*32,16),
        text=rand(S,H),vision=rand(S//2,H),temporal_w=weight(H,H*2),
        mask=(torch.arange(S)%2==0),pos=torch.arange(S,dtype=torch.float32),
        mamba_in=weight(2*H+HEADS,H),dt_w=weight(16,16),dt_x=rand(S,16),
        dt_bias=rand(16),A_log=rand(16),B=rand(S,4,8),
        eq_gate=weight(16*I,H).view(16,I,H),eq_up=weight(16*I,H).view(16,I,H),
        eq_down=weight(16*H,I).view(16,H,I),
        q=rand(HEADS,S,D),k=rand(KVHEADS,S,D),v=rand(KVHEADS,S,D),
        fp4_levels=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.]),
        fp4_midpoints=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.]))

class Control:
    def __init__(self,number,inputs,mode,*,fib=False,cpu_reference=False):
        self.number=number;self.p=inputs;self.mode=mode;self.fib=fib;self.cpu_reference=cpu_reference
    def _round_fp4(self,x):
        # E2M1 codebook with nearest-even ties (uniform half-step is NOT FP4).
        a=x.abs().clamp(max=6)
        if self.cpu_reference:
            # Independent nearest-code search for reference, including ties.
            levels=self.p['fp4_levels']; dist=(a[...,None]-levels).abs()
            minimum=dist.amin(-1,keepdim=True); matches=dist==minimum
            idx=matches.to(torch.int64).argmax(-1)
            up=(idx+1).clamp(max=7)
            tied=matches.gather(-1,up[...,None]).squeeze(-1)&(idx<7)
            idx=torch.where(tied&((idx%2)==1),up,idx)
        else:
            mids=self.p['fp4_midpoints'];idx=torch.bucketize(a.contiguous(),mids)
            at=mids[idx.clamp(max=6)]
            idx=idx+((idx<7)&(a==at)&((idx%2)==1)).long()
        return self.p['fp4_levels'][idx]*x.sign()
    def quant(self,x,weight=False):
        x=x.float()
        # Activations: 1x128 for E4M3, 1x16 for E2M1. FP8 weights:
        # 128x128; reduced edge blocks are scaled over their actual elements.
        if self.mode=='bf16':return x.to(torch.bfloat16).float()
        block=16 if self.mode=='fp4' else 128
        if weight and self.mode=='fp8' and x.shape[-2]<=128 and x.shape[-1]<=128:
            scale=x.abs().amax((-2,-1),keepdim=True).clamp_min(1e-12)/448
            return (x/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale
        k=x.shape[-1]; pad=(-k)%block
        z=F.pad(x,(0,pad)).reshape(*x.shape[:-1],-1,block)
        maximum=6 if self.mode=='fp4' else 448
        global_scale=(x.abs().amax().clamp_min(1e-12)/(448*6)) if self.mode=='fp4' else x.new_tensor(1.)
        scale=(z.abs().amax(-1,keepdim=True).clamp_min(1e-12)/(maximum*global_scale))
        if self.mode=='fp4':scale=scale.clamp(min=2**-9,max=448).to(torch.float8_e4m3fn).float()
        scale=scale*global_scale
        normalized=(z/scale).clamp(-maximum,maximum)
        q=self._round_fp4(normalized) if self.mode=='fp4' else normalized.to(torch.float8_e4m3fn).float()
        return (q*scale).reshape(*x.shape[:-1],-1)[...,:k]
    def linear(self,x,w,bias=None):
        # A software-dequantized FP32 GEMM; native FP8/NVFP4 instructions absent.
        y=self.quant(x)@self.quant(w,weight=True).transpose(-1,-2)
        if bias is not None:y=y+bias
        return y.to(torch.bfloat16).float()
    @staticmethod
    def norm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
    def rotate(self,x):
        d=x.shape[-1];inv=10000.**(-torch.arange(0,d,2,device=x.device).float()/d)
        f=self.p['pos'][:,None]*inv; c=f.cos().repeat(1,2);s=f.sin().repeat(1,2)
        a,b=x.chunk(2,-1)
        return x*c+torch.cat((-b,a),-1)*s
    def qkv(self,x,norm=False,rope=False):
        q=self.linear(x,self.p['wq']).view(S,HEADS,D).transpose(0,1)
        k=self.linear(x,self.p['wk']).view(S,KVHEADS,D).transpose(0,1)
        v=self.linear(x,self.p['wv']).view(S,KVHEADS,D).transpose(0,1)
        if norm:q,k=self.norm(q),self.norm(k)
        if rope:q,k=self.rotate(q),self.rotate(k)
        return q,k,v
    def attention(self,q,k,v,quantized_scores=False):
        k=k.repeat_interleave(HEADS//KVHEADS,dim=0);v=v.repeat_interleave(HEADS//KVHEADS,dim=0)
        q,k=(self.quant(q),self.quant(k)) if quantized_scores else (q,k)
        score=(q@k.transpose(-1,-2))/math.sqrt(D)
        mask=torch.ones(S,S,device=q.device,dtype=torch.bool).triu(1)
        p=score.masked_fill(mask,-torch.inf).softmax(-1)
        if quantized_scores:p,v=self.quant(p),self.quant(v)
        return p@v,p
    def mlp(self,x,gate=None,up=None,down=None):
        p=self.p;g=self.linear(x,p['gate'] if gate is None else gate);u=self.linear(x,p['up'] if up is None else up)
        z=F.silu(g)*u
        return self.linear(z,p['down'] if down is None else down)
    def group_route(self,logits,groups=4,group_k=2,topk=2):
        # Scores for selecting groups include correction bias; final weights
        # use original sigmoid scores, matching DeepSeek-style routing.
        p=logits.sigmoid();s=p+self.p['route_bias']
        per=logits.shape[-1]//groups
        group_score=s.reshape(S,groups,per).topk(min(2,per),-1).values.sum(-1)
        chosen=group_score.topk(group_k,-1).indices
        keep=torch.zeros_like(group_score).scatter(1,chosen,1).repeat_interleave(per,1).bool()
        ids=s.masked_fill(~keep,-torch.inf).topk(topk,-1).indices
        weights=p.gather(1,ids);weights=weights/weights.sum(-1,keepdim=True)
        return weights,ids
    def __call__(self):
        p=self.p;x=p['x'];n=self.number
        if self.fib:
            weights,ids=self.group_route(self.linear(x,p['router']),groups=8,group_k=4,topk=8)
            # Four local experts from a 16-expert global routing population.
            # Apply the actual selected expert IDs; never ignore routing.
            y=torch.zeros_like(x)
            for e in range(4):
                w=(weights*(ids==e)).sum(-1,keepdim=True)
                y=y+w*self.mlp(x,p['eq_gate'][e],p['eq_up'][e],p['eq_down'][e])
            return y
        if n in (1,6):return self.linear(x,p['wo'],p['bias'])
        if n in (10,15,20,22,24,26):return self.linear(x,p['wo'])
        if n==2:return self.qkv(x,norm=True)
        if n in (7,25):return self.qkv(x)
        if n==3:return F.silu(self.linear(x,p['gate']))*self.linear(x,p['up'])
        if n==4:return self.mlp(x)*p['routing_weight']
        if n in (5,27):return self.linear(x,p['router'])
        if n==8:
            projected=self.linear(p['vision'],p['wo'])
            return p['text'].index_copy(0,p['mask'].nonzero().flatten(),projected)
        if n==9:return self.linear(x.reshape(S//2,2*H),p['temporal_w'])
        if n==11:return self.group_route(self.linear(x,p['router']))
        if n in (12,17,29,32):return self.mlp(x)
        if n==13:
            kv=self.linear(x,p['kv_a']);latent,pe=kv.chunk(2,-1)
            expanded=self.linear(self.norm(latent),p['kv_b']).view(S,HEADS,32)
            return expanded,pe
        if n==14:
            dim=64;base=10000.;factor=40.;maxpos=4096
            inv=base**(-torch.arange(0,dim,2,device=x.device).float()/dim)
            lo=math.floor(dim*math.log(maxpos/(32*2*math.pi))/(2*math.log(base)))
            hi=math.ceil(dim*math.log(maxpos/(1*2*math.pi))/(2*math.log(base)))
            ramp=((torch.arange(dim//2,device=x.device)-max(0,lo))/max(1,min(dim-1,hi)-max(0,lo))).clamp(0,1)
            freq=self.p['pos'][:,None]*(inv*(1-ramp)+inv/factor*ramp)
            f=self.quant(freq);f=torch.cat((f,f),-1)
            return f.cos(),f.sin()
        if n==16:
            q=self.linear(self.norm(self.linear(x,p['q_a'])),p['q_b'])
            latent,pe=self.linear(x,p['kv_a']).chunk(2,-1)
            kv=self.linear(self.norm(latent),p['kv_b']).view(S,HEADS,32)
            k,v=kv.chunk(2,-1)
            return q.view(S,HEADS,D),k,v,pe
        if n==18:return self.linear(x,p['wo'])+p['residual']
        if n in (19,30):
            q,k,v=self.qkv(self.norm(x) if n==30 else x,norm=n==30,rope=True)
            y,_=self.attention(q,k,v)
            return self.linear(y.transpose(0,1).reshape(S,H),p['wo'])
        if n==21:return self.linear(x,p['mamba_in'])
        if n==23:
            dt=F.softplus(self.linear(p['dt_x'],p['dt_w'],p['dt_bias'])).clamp(.001,1.)
            da=torch.exp(dt*(-p['A_log'].exp()))
            db=(dt.view(S,4,4,1)*p['B'][:,:,None,:]).reshape(S,16,8)
            return dt,da,db
        if n==28:return self.qkv(x,rope=True)
        if n==31:return self.attention(p['q'],p['k'],p['v'],quantized_scores=True)
        if n==33:
            logits=self.linear(x,p['router']);weights,ids=logits.softmax(-1).topk(2,-1)
            return weights/weights.sum(-1,keepdim=True),ids,logits
        raise ValueError(n)

def flatten(o):return list(o) if isinstance(o,(tuple,list)) else [o]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--slug',required=True);ap.add_argument('--kind',help='Legacy hint ignored: dispatch is by row ID, not substring')
    ap.add_argument('--mode',choices=['fp8','fp4','bf16']);ap.add_argument('--seed',type=int,default=120509);ap.add_argument('--check',action='store_true');ap.add_argument('--bench',type=int,default=0)
    a=ap.parse_args();num=int(a.slug[:3]);fib='block_scale_ds_routing' in a.slug
    mode=a.mode or ('fp8' if fib or num not in FP4_IDS else 'fp4')
    torch.set_num_threads(2);torch.manual_seed(a.seed);torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability()==(8,6)
    torch.backends.cuda.matmul.allow_tf32=False
    cpu=make_inputs(a.seed);p={k:v.cuda() for k,v in cpu.items()}
    op=Control(num,p,mode,fib=fib)
    if a.check:
        expected=Control(num,cpu,mode,fib=fib,cpu_reference=True)();actual=op();torch.cuda.synchronize()
        errors=[]
        for got,ref in zip(flatten(actual),flatten(expected)):
            got=got.cpu()
            if not ref.is_floating_point():torch.testing.assert_close(got,ref,rtol=0,atol=0);continue
            torch.testing.assert_close(got,ref,rtol=.03,atol=.003)
            errors.append(float((got-ref).abs().max()))
        print(json.dumps({'slug':a.slug,'check':'CPU reference and GPU outputs agree','max_absolute_errors':errors,'seed':a.seed}))
        return
    # Warm initialization without recording: reuse exactly the same operands.
    op();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    desc={'slug':a.slug,'operation':'group_limited_local_expert_moe' if fib else OPS[num],
          'mode':mode,'implementation':'sm86_row_specific_software_quant_v2','geometry':{'S':S,'H':H,'I':I,'query_heads':HEADS,'kv_heads':KVHEADS,'head_dim':D},
          'native_low_precision_mma':False,'input_bytes':sum(v.numel()*v.element_size() for v in p.values()),
          'quantization':'software E4M3 block128 / E2M1 block16 with FP8 block scales; FP32 GEMM',
          'seed':a.seed,'torch':torch.__version__}
    dest=os.getenv('SM86_RUN_ARTIFACT_DIR') or os.getenv('TRACES_FOLDER')
    if dest:
        d=Path(dest)/'workload_source';d.mkdir(parents=True,exist_ok=True)
        (d/'quant_substitute_runner.py').write_text(Path(__file__).read_text());(d/'run.json').write_text(json.dumps(desc,indent=2)+'\n')
    if a.bench:
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        for _ in range(a.bench):out=op()
        end.record();torch.cuda.synchronize();desc['cuda_ms']=start.elapsed_time(end)/a.bench
    else:
        torch.cuda.cudart().cudaProfilerStart()
        out=op();torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    assert all(bool(torch.isfinite(v).all()) for v in flatten(out) if v.is_floating_point())
    desc.update(output_shapes=[list(v.shape) for v in flatten(out)],peak_allocated_bytes=torch.cuda.max_memory_allocated(),finite=True)
    print(json.dumps(desc,sort_keys=True))
if __name__=='__main__':main()
