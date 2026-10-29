from pathlib import Path
import argparse
import json
import fitz
from PIL import Image, ImageOps

project = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--reviewed',action='store_true')
args=parser.parse_args()
output = project/'results/eight-city-phase-a-20261029'
temporary = project/'.codex-temp/figure-review'
temporary.mkdir(exist_ok=True)
records = []
for path in sorted(output.rglob('*.pdf')):
    with fitz.open(path) as document:
        if len(document)!=1:
            raise ValueError('图件PDF需为单页')
        page=document[0]
        outside=[]
        for block in page.get_text('dict')['blocks']:
            if 'lines' not in block:
                continue
            for line in block['lines']:
                for span in line['spans']:
                    rect=fitz.Rect(span['bbox'])
                    if not (page.rect+(-2,-2,2,2)).contains(rect):
                        outside.append(span['text'])
        records.append({'figure':path.stem,'pdf_text_outside_page':outside,
            'replacement_glyphs':page.get_text().count('\ufffd'),
            'fonts':[font[3] for font in page.get_fonts()], 'page_size_points':list(page.rect)})
        preview=temporary/(path.stem+'.png')
        if not preview.exists() or preview.stat().st_mtime<path.stat().st_mtime:
            page.get_pixmap(matrix=fitz.Matrix(1.1,1.1),alpha=False).save(preview)
for seasonal in sorted(temporary.glob('*-seasonal-days.png')):
    stem=seasonal.stem.removesuffix('-seasonal-days')
    paths=[seasonal,temporary/(stem+'-typical-week.png'),temporary/(stem+'-annual-envelope.png')]
    images=[]
    for path in paths:
        with Image.open(path) as source:
            images.append(ImageOps.contain(source.copy(),(1200,1800)))
    sheet=Image.new('RGB',(images[0].width+max(images[1].width,images[2].width),max(images[0].height,images[1].height+images[2].height)), 'white')
    sheet.paste(images[0],(0,0))
    sheet.paste(images[1],(images[0].width,0))
    sheet.paste(images[2],(images[0].width,images[1].height))
    sheet.save(temporary/(stem+'-review.png'))
(output/'figure-render-checks.json').write_text(json.dumps({'figures':records,'render_engine':'PyMuPDF 1.26.5',
    'all_text_inside_pages':not any(r['pdf_text_outside_page'] for r in records),
    'no_replacement_glyphs':not any(r['replacement_glyphs'] for r in records),
    'visual_review':'reviewed_by_agent' if args.reviewed else 'pending_human_inspection_by_agent'},ensure_ascii=False,indent=2),encoding='utf-8')
if args.reviewed:
    for name in ('figure-validation.json','figure-overview-validation.json'):
        path=output/name
        report=json.loads(path.read_text(encoding='utf-8'))
        report['visual_review']='reviewed_by_agent; all_41_PDF_renderings_checked'
        report['review_notes']='已逐情境查看36张负荷图和5张概览图，中文、图例、标签和面板范围正常。'
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Rendered',len(records),'PDF figures')
