# Export llama2.c stories260K checkpoint to int8 (per-row) weights embedded in a PostScript file.
import struct, sys, array
ckpt, tokf, out = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(ckpt,'rb').read()
dim, hid, L, nh, nkv, V, seq = struct.unpack('7i', raw[:28])
shared = V > 0; V = abs(V)
f = array.array('f'); f.frombytes(raw[28:])
hs = dim//nh; kvd = dim*nkv//nh
p = 0
def take(n):
    global p
    a = f[p:p+n]; p += n; return a
emb = take(V*dim); rms_att = take(L*dim); wq = take(L*dim*dim); wk = take(L*dim*kvd); wv = take(L*dim*kvd)
wo = take(L*dim*dim); rms_ffn = take(L*dim); w1 = take(L*hid*dim); w2 = take(L*dim*hid); w3 = take(L*hid*dim)
rms_f = take(dim)
assert shared
nparams = V*dim + L*(2*dim*dim + 2*dim*kvd + 3*dim*hid + 2*dim) + dim
o = open(out,'w')
def psarr(items, chunk=200):
    # build arrays in chunks so the operand stack never exceeds ~200 items (Adobe limit is 500)
    if len(items) <= chunk: return '[' + ' '.join(items) + ']'
    out = '%d array' % len(items)
    for i in range(0, len(items), chunk):
        out += ' dup %d [%s] putinterval\n' % (i, ' '.join(items[i:i+chunk]))
    return out
def realarr(a): return psarr(['%.6g' % v for v in a])
def qrows(a, rows, cols):
    # returns PS text: [ [scales] [<hex> ...] ]
    sc=[]; hx=[]
    for r in range(rows):
        row = a[r*cols:(r+1)*cols]
        m = max(abs(v) for v in row) or 1e-8
        s = m/127.0
        b = bytes(max(1,min(255,int(round(v/s))+128)) for v in row)
        sc.append(s); hx.append('<'+b.hex()+'>')
    return '[' + realarr(sc) + '\n' + psarr(hx) + ']'
o.write('%!PS-Adobe-3.0\n%%Title: TONER-1 (stories260K int8, PostScript inference)\n')
o.write('/D %d def /HID %d def /NL %d def /NH %d def /NKV %d def /V %d def /SEQ %d def /HS %d def /KVD %d def /KVMUL %d def /NPARAMS %d def\n' % (dim,hid,L,nh,nkv,V,seq,hs,kvd,nh//nkv,nparams))
o.write('/EMB ' + qrows(emb, V, dim) + ' def\n')
for name, t, r, c in [('WQ',wq,dim,dim),('WK',wk,kvd,dim),('WV',wv,kvd,dim),('WO',wo,dim,dim),('W1',w1,hid,dim),('W2',w2,dim,hid),('W3',w3,hid,dim)]:
    o.write('/%s [\n' % name)
    for l in range(L):
        o.write(qrows(t[l*r*c:(l+1)*r*c], r, c) + '\n')
    o.write('] def\n')
o.write('/RMSATT [' + ' '.join(realarr(rms_att[l*dim:(l+1)*dim]) for l in range(L)) + '] def\n')
o.write('/RMSFFN [' + ' '.join(realarr(rms_ffn[l*dim:(l+1)*dim]) for l in range(L)) + '] def\n')
o.write('/RMSF ' + realarr(rms_f) + ' def\n')
# tokenizer: decode pieces into raw bytes
tb = open(tokf,'rb').read(); q = 4; pieces=[]
for i in range(V):
    sc, ln = struct.unpack('fi', tb[q:q+8]); q += 8
    s = tb[q:q+ln]; q += ln
    if len(s)==6 and s.startswith(b'<0x') and s.endswith(b'>'):
        s = bytes([int(s[3:5],16)])
    pieces.append(s)
o.write('/VOCAB ' + psarr(['<'+s.hex()+'>' for s in pieces]) + ' def\n')
o.close()
print('params', nparams, 'dim', dim, 'hid', hid, 'L', L, 'nh', nh, 'nkv', nkv, 'V', V, 'seq', seq, 'used floats', p, 'of', len(f))
