import hw4 as h
from pathlib import Path
f=Path('public_test'); docs=h.load_json(f/'docs.json'); gold=h.load_json(f/'ground_truth.json'); qs=h.load_json(f/'questions.json')
chunks=[]
for name,html in h.load_corpus(f/'corpus').items():
 c=h.chunk_document(name,html); p=h.document_problems(c,None,docs[name],name)
 if p: print('DOC',name,p)
 chunks+=c
print('TABLE',[h.table_score(h.table_to_text(t['html']),t) for t in h.load_json(f/'tables.json')])
i=h.build_index(chunks); hits=kept=rej=0; ps=0
for qid,q in qs.items():
 g=gold[qid]; p=h.parse_query(q); r=h.retrieve(i,q); ps+=h.parse_score(p,g)
 if h.parse_score(p,g)<1: print('PARSE',qid,p,g['query'])
 if g['evidence']:
  kept+=bool(r); hits+=h.retrieval_hit(r,g)
  if not h.retrieval_hit(r,g): print('MISS',qid,q,[(c.metadata['source'],c.page_content[:120]) for c in r])
 else:
  rej+=not r
  if r: print('FALSE KEEP',qid)
print('chunks',len(chunks),'parse',ps/60,'hits',hits,'kept',kept,'rejected',rej)
