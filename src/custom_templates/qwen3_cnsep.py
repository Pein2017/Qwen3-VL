from typing import List, Literal

from swift.llm.template.template.qwen import Qwen3VLTemplate, QwenTemplateMeta
from swift.llm.template.register import register_template
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context


class Qwen3VLCnSepTemplate(Qwen3VLTemplate):
    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        tokens = super().replace_tag(media_type, index, inputs)
        if media_type == 'image':
            return [f'图片{index + 1}', *tokens]
        if media_type == 'video':
            return [f'视频{index + 1}', *tokens]
        return tokens


# Register a unique template key for YAML selection
register_template(QwenTemplateMeta('qwen3_vl_cnsep', template_cls=Qwen3VLCnSepTemplate, default_system=None))


