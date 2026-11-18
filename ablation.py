"""One-component-at-a-time experiments; annotations are used ONLY for scoring."""
import argparse
import csv
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import hw4 as h

VARIANTS = ['B0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
CHANGES = ['Naive baseline', 'Serialize tables', 'Section-aware chunks', 'Normalize lexical tokens', 'Filter company/type/year', 'Remove company tokens', 'Boost table term coverage', 'Allocate slots per company', 'Reject unsupported queries']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--generate',action='store_true'); ap.add_argument('--data-folder',default='public_test'); ap.add_argument('--variants',nargs='+',default=VARIANTS); a=ap.parse_args()
    folder=Path(a.data_folder); out=Path('experiments'); out.mkdir(exist_ok=True)
    corpus=h.load_corpus(folder/'corpus'); qs=h.load_json(folder/'questions.json'); gold=h.load_json(folder/'ground_truth.json')
    splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)
    summaries=[]
    for v in a.variants:
        n=VARIANTS.index(v); start=time.perf_counter(); chunks=[]
        for source,html in corpus.items():
            if n>=2: chunks.extend(h.chunk_document(source,html)); continue
            soup=BeautifulSoup(html,'html.parser'); meta=h._metadata(source,soup)
            if n>=1:
                soup=BeautifulSoup(html,'html.parser')
                for t in soup.find_all('table'): t.replace_with(h.table_to_text(str(t)))
            chunks.extend(h.Document(page_content=t,metadata=meta) for t in splitter.split_text(soup.get_text(' ',strip=True)))
        index=h.build_index(chunks)
        if n<3: index['bm25']=BM25Okapi([c.page_content.lower().split() for c in chunks])
        retrieved={}; parsed={}
        for qid,q in qs.items():
            p=h.parse_query(q); parsed[qid]=p
            if n==8: retrieved[qid]=h.retrieve(index,q); continue
            if n>=5:
                rankings=[h._rank(index,q,p,c,boost=n>=6) for c in p['companies']]
                if n==7:
                    ids=[]
                    for depth in range(5):
                        for r in rankings:
                            if depth<len(r) and len(ids)<5: ids.append(r[depth][1])
                else: ids=[i for _,i in sorted([x for r in rankings for x in r],reverse=True)[:5]]
            else:
                tokens=q.lower().split() if n<3 else h._tokens(q)
                scores=index['bm25'].get_scores(tokens); ids=list(range(len(chunks)))
                if n==4:
                    ids=[i for i,c in enumerate(chunks) if h._key(c.metadata['company']).lower() in {h._key(x).lower() for x in p['companies']} and c.metadata['doc_type']==p['doc_type'] and (p['fiscal_year'] is None or c.metadata['fiscal_year'] in ({p['fiscal_year'],p['fiscal_year']+1} if p['doc_type']=='annual_report' else {p['fiscal_year'],p['fiscal_year']-1,p['fiscal_year']-2}))]
                ids=sorted(ids,key=lambda i:(-scores[i],i))[:5]
            retrieved[qid]=[chunks[i] for i in ids]
        answers={}
        if a.generate:
            with ThreadPoolExecutor(max_workers=8) as pool:
                answers=dict(zip(qs,pool.map(lambda qid:h.generate(qs[qid],retrieved[qid]),qs)))
        rows=[]
        for qid,q in qs.items():
            g=gold[qid]; cs=retrieved[qid]
            rows.append(dict(id=qid,category=g['category'],question=q,parsed=json.dumps(parsed[qid]),parse_score=h.parse_score(parsed[qid],g) if n>=4 else '',retrieved=';'.join(sorted({c.metadata['source'] for c in cs})),valid=True,retrieval_hit=h.retrieval_hit(cs,g) if g['evidence'] else '',answer=answers.get(qid,''),correct=h.answer_correct(answers[qid],g) if a.generate else '',expected=g['answer']))
        with (out/f'{v}_results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        ans=[qid for qid,g in gold.items() if g['evidence']]; un=[qid for qid in qs if qid not in ans]
        result=dict(variant=v,change=CHANGES[n],chunks=len(chunks),recall=sum(h.retrieval_hit(retrieved[qid],gold[qid]) for qid in ans)/len(ans),kept=sum(bool(retrieved[qid]) for qid in ans)/len(ans),rejected=sum(not retrieved[qid] for qid in un)/len(un),accuracy=sum(h.answer_correct(answers[qid],gold[qid]) for qid in qs)/len(qs) if a.generate else None,seconds=round(time.perf_counter()-start,2))
        summaries.append(result); print(result,flush=True)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    print('Generator usage:',h.GENERATOR_USAGE.summary())
if __name__=='__main__': main()
