# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from .extract_themes import PROMPT as EXTRACT_THEMES_PROMPT
from .label_utterances import PROMPT as LABEL_UTTERANCES_PROMPT
from .label_clusters import PROMPT as LABEL_CLUSTERS_PROMPT
from .label_clusters_multi import PROMPT as LABEL_CLUSTERS_MULTI_PROMPT
from .label_clusters_multi import PROMPT_CONCLUDE_RULE as LABEL_CONCLUDE_PROMPT
from .label_clusters_multi import PROMPT_FILTER_RULE as LABEL_FILTER_PROMPT

from .styleguide import (
    SECTION_1_PROMPT as STYLEGUIDE_SECTION_1_PROMPT,
    SECTION_2_PROMPT as STYLEGUIDE_SECTION_2_PROMPT,
    SECTION_3_PROMPT as STYLEGUIDE_SECTION_3_PROMPT
)