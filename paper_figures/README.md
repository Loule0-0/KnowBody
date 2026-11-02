# 当前论文 Fig.2 / Fig.3：可编辑 PowerPoint

| 图 | 可编辑 PPT | 论文固定版式 | 预览 |
| --- | --- | --- | --- |
| Fig.2：整体闭环与结构化经验库 | [下载 PPT](Fig2_KnowBody_Harness_Editable.pptx?raw=true) | [PDF](Fig2_KnowBody_Harness_Paper.pdf) | [PNG](Fig2_Preview.png) |
| Fig.3：身体关系与外部交互案例 | [下载 PPT](Fig3_Body_External_Editable.pptx?raw=true) | [PDF](Fig3_Body_External_Paper.pdf) | [论文预览](Fig3_Paper_Preview.png) · [PPT实际导出](Fig3_PPT_Preview.png) |

两份论文 PDF 分别与当前 LaTeX 引用的 `fig9_harness.pdf`、`figures/fig3_body_relations.pdf` 字节一致。

两份 PPT 已按手工编辑需要重新分组，而非把底层绘图对象平铺在幻灯片上。Fig.2 在原生 PPT 上整理，Fig.3 从当前矢量图场景重建；保留原文字、照片裁切和标注位置。

| 文件 | 顶层组织 | 组内编辑单位 |
| --- | --- | --- |
| Fig.2 | **5 个命名组**：交互场景、执行闭环、记录箭头、执行架构、经验库 | 执行模型、检索、管理模型、输入卡片、身体/外部经验行、E/C/K 卡片、历史条带等分别分组 |
| Fig.3 | **2 个命名组**：Body relations、External interactions | 五个案例面板分别成组；照片连同标注和标题、完整箭头、知识说明块各自成组 |

文字保留为完整的标题、句子或说明段文本框，没有拆成逐字轮廓。Fig.2 有 72 个非空文字框、23 张照片；Fig.3 有 35 个文字框、9 处原照片。图片使用原始素材及可调整裁切，标注可独立修改；复合箭头及图标的组成部件收在有语义的组内，不会作为散件出现在顶层选择列表。

## 手工修改

1. **移动或缩放模块**：在“选择窗格”选择相应命名组，整组调整；不需要框选几十个零件。
2. **改一处文字**：先选中组，再点组内文字框直接编辑，无需把整图解组。
3. **替换照片**：进入照片组，选中图片使用“更改图片”，再按需要调整裁切；边框、标题、标注保持各自可编辑。
4. **调整箭头或案例**：选对应完整箭头组或案例卡片组；需要改内部细节时再进入该组。

最多四层语义分组。原先 243 / 96 个底层对象分别保留在 Fig.2 / Fig.3 的对应组内；这些对象用于精细修改，正常编辑从五组 / 两组入口展开。可看 [完整分组清单与检查](EDITABILITY_CHECK.json)。

已检查原文字、图片字节、裁切和几何，并将整理前后的 PPT 通过 LibreOffice 导出对照。论文精确版式仍由配套 PDF 提供，其他机器字体替换可能有轻微字宽差异。

Fig.3 的 (d) 为用户提供的抓持示范候选，(e) 为仿真记录中的保持/失持观察。修改图时保留这两类证据身份，不把示范画成已验证的自主成功。

`source/fig3/assets/` 保留原照片和软蒙版；`scene.py` 为当前矢量图的场景布局；`build_pptx.py` 是 PPT 重建工具，生成后自动应用 `source/organize_for_editing.py` 中的语义分组。需要重新生成时安装 `python-pptx==1.0.2`、Pillow、ReportLab，并提供自己具有使用权的 Arial 常规/粗体 TTF：

```bash
python source/fig3/build_pptx.py \
  --font-regular /path/to/Arial.TTF --font-bold /path/to/Arialbd.TTF \
  --output /path/to/Fig3_Editable.pptx
```

不用重新生成即可用 PowerPoint 打开并逐组修改。`FIGURE_MANIFEST.json` 记录源文件对应关系及检查范围；本目录不含字体文件。
