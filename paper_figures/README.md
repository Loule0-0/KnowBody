# 当前论文 Fig.2 / Fig.3：可编辑 PowerPoint

| 图 | 可编辑 PPT | 论文固定版式 | 预览 |
| --- | --- | --- | --- |
| Fig.2：整体闭环与结构化经验库 | [下载 PPT](Fig2_KnowBody_Harness_Editable.pptx?raw=true) | [PDF](Fig2_KnowBody_Harness_Paper.pdf) | [PNG](Fig2_Preview.png) |
| Fig.3：身体关系与外部交互案例 | [下载 PPT](Fig3_Body_External_Editable.pptx?raw=true) | [PDF](Fig3_Body_External_Paper.pdf) | [论文预览](Fig3_Paper_Preview.png) · [PPT实际导出](Fig3_PPT_Preview.png) |

两份论文 PDF 分别与当前 LaTeX 引用的 `fig9_harness.pdf`、`figures/fig3_body_relations.pdf` 字节一致。

Fig.2 直接保留原生 PPT。Fig.3 原来由矢量脚本生成，这里按相同场景坐标、文字、原照片及标注重建为 **96 个独立对象，35 处可编辑文字**。文本框、面板、箭头、圆标注和照片裁切均可编辑；照片本身保留真实素材，不把整张论文图作为一张图片塞入 PPT。已把 PPT 导出为 PDF 并直接看图对照，检查五个面板、文字、图像和箭头；字体替换可能产生轻微字宽差异，论文精确版式以配套 PDF 为准。

Fig.3 的 (d) 为用户提供的抓持示范候选，(e) 为仿真记录中的保持/失持观察。修改图时保留这两类证据身份，不把示范画成已验证的自主成功。

`source/fig3/assets/` 保留原照片和软蒙版；`scene.py` 为当前矢量图的场景布局；`build_pptx.py` 是独立的 PPT 重建工具。需要重新生成时安装 `python-pptx==1.0.2`、Pillow、ReportLab，并提供自己具有使用权的 Arial 常规/粗体 TTF：

```bash
python source/fig3/build_pptx.py \
  --font-regular /path/to/Arial.TTF --font-bold /path/to/Arialbd.TTF \
  --output /path/to/Fig3_Editable.pptx
```

不用重新生成也可直接用 PowerPoint 打开、解组或选择对象修改。`FIGURE_MANIFEST.json` 记录源文件对应关系及检查范围；本目录不含字体文件。
