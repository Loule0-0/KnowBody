"""Rebuild Fig.3 as native PowerPoint objects from its paper scene and real photos."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import math
from PIL import Image
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import MSO_ANCHOR
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font-regular',required=True,type=Path)
parser.add_argument('--font-bold',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
OUT=Path(__file__).resolve().parents[2]
ASSETS=Path(__file__).resolve().parent/'assets'
inputs=json.loads((ASSETS.parent/'INPUTS.json').read_text())
for row in inputs['photos']:
    assert hashlib.sha256((ASSETS/row['file']).read_bytes()).hexdigest()==row['sha256']
for name,path in [('Arial',args.font_regular),('ArialBold',args.font_bold)]:
    pdfmetrics.registerFont(TTFont(name,str(path)))
scene=(ASSETS.parent/'scene.py').read_text()
deck=Presentation();scale=16.11*72/504
deck.slide_width=Pt(504*scale);deck.slide_height=Pt(312*scale)
slide=deck.slides.add_slide(deck.slide_layouts[6]);shapes=[];texts=[]


def unit(n):return Pt(n*scale)
def rgb(c):return RGBColor.from_string(c.lstrip('#'))
def clean(shape,label):
    shape.name=label
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    for e in shape._element.findall('.//a:effectRef',ns):e.set('idx','0')
    shapes.append(shape)
    return shape
def box(x,y,w,h,fill='#FFFFFF',stroke='#CBD4DC',radius=4,lw=.7):
    s=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,unit(x),unit(y),unit(w),unit(h))
    s.adjustments[0]=min(.45,radius/min(w,h))
    s.fill.solid();s.fill.fore_color.rgb=rgb(fill)
    s.line.color.rgb=rgb(stroke);s.line.width=unit(lw)
    return clean(s,'Editable panel / label background')
def text(x,y,value,size=8,color='#20313E',bold=False):
    width=pdfmetrics.stringWidth(value,'ArialBold' if bold else 'Arial',size)+3
    s=slide.shapes.add_textbox(unit(x),unit(y-.4),unit(width),unit(size*1.5))
    f=s.text_frame;f.clear();f.margin_left=f.margin_right=f.margin_top=f.margin_bottom=0
    f.word_wrap=False;f.vertical_anchor=MSO_ANCHOR.TOP
    p=f.paragraphs[0];p.text=value;p.space_before=p.space_after=0
    p.font.name='Arial';p.font.size=unit(size);p.font.bold=bold;p.font.color.rgb=rgb(color)
    texts.append(value)
    return clean(s,'Editable text: '+value)
def polygon(points,color):
    pts=[(int(unit(x)),int(unit(y))) for x,y in points]
    b=slide.shapes.build_freeform(*pts[0]);b.add_line_segments(pts[1:],close=True);s=b.convert_to_shape()
    s.fill.solid();s.fill.fore_color.rgb=rgb(color);s.line.fill.background()
    return clean(s,'Editable arrowhead')
def line(points,color='#477BB8',width=1,dash=False,arrow=False):
    for (x,y),(u,v) in zip(points,points[1:]):
        s=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,unit(x),unit(y),unit(u),unit(v))
        s.line.color.rgb=rgb(color);s.line.width=unit(width)
        if dash:s.line.dash_style=MSO_LINE_DASH_STYLE.DASH
        clean(s,'Editable connector')
    if arrow:
        x,y=points[-2];u,v=points[-1];a=math.atan2(v-y,u-x)
        polygon([(u,v),*((u-4*math.cos(a+off),v-4*math.sin(a+off)) for off in [-.46,.46])],color)
def circle(x,y,r,stroke,width,fill=None):
    s=slide.shapes.add_shape(MSO_SHAPE.OVAL,unit(x-r),unit(y-r),unit(2*r),unit(2*r))
    if fill:s.fill.solid();s.fill.fore_color.rgb=rgb(fill)
    else:s.fill.background()
    s.line.color.rgb=rgb(stroke);s.line.width=unit(width)
    return clean(s,'Editable observation marker')
def point(x,y,color,r=2.2):return circle(x,y,r,'#FFFFFF',.8,color)
def ring(x,y,color,r=4):
    circle(x,y,r,'#FFFFFF',2);circle(x,y,r,color,.85)
def photo(path,x,y,w,h,roi=(0,0,1,1),restore_mask=None):
    path=Path(path);im=Image.open(path);iw,ih=im.size;source=path
    if restore_mask:
        im=im.convert('RGBA');im.putalpha(Image.open(restore_mask).convert('L'));buffer=io.BytesIO();im.save(buffer,format='PNG');buffer.seek(0);source=buffer
    l,t,r,b=roi;s=max(w/((r-l)*iw),h/((b-t)*ih))
    x0=x+w/2-(l+r)*iw*s/2;y0=y+h/2-(t+b)*ih*s/2
    pic=slide.shapes.add_picture(str(source) if isinstance(source,Path) else source,unit(x),unit(y),width=unit(w),height=unit(h))
    pic.crop_left=(x-x0)/(iw*s);pic.crop_right=1-pic.crop_left-w/(iw*s)
    pic.crop_top=(y-y0)/(ih*s);pic.crop_bottom=1-pic.crop_top-h/(ih*s)
    clean(pic,'Original photo with editable viewport crop: '+path.name)
    return lambda u,v:(x0+u*s,y0+v*s)
def tag(x,y,value,color='#477BB8'):
    width=pdfmetrics.stringWidth(value,'Arial',7.5)+8
    box(x,y,width,11,'#FFFFFF',color,2,.5);text(x+4,y+1,value,7.5,color)

ns=globals().copy();ns.update(O=ASSETS.parent,F=ASSETS,B=ASSETS,
    INK='#20313E',BLUE='#477BB8',ORANGE='#C97835',GREEN='#5E8C43',GRAY='#CBD4DC',MUTED='#586774')
exec(compile(scene,str(OUT/'source/fig3/scene.py'),'exec'),ns)
target=args.output;deck.save(target)
assert len(slide.shapes)>80 and len(texts)==len(inputs['texts'])
assert texts==inputs['texts']
print('PASS: saved',target,'native shapes',len(slide.shapes),'editable text',len(texts))
