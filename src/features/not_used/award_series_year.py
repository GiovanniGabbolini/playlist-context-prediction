"""
Created on --

@author Name Redacted Surname Redacted
"""

import re

from src.utils.decorator_annotations import annotations


@annotations({'entailed': True})
def award_series_year(award_wikidata) -> 'award_series_year':
    """Extracts a tuple from artist awards dictionary.
    
    The edge produced by this feature is not counted in interestingess's shortness heuristics
    """
    if 'award_series' in award_wikidata['value']:
        assert re.match(r"^\d{4}$", award_wikidata['value']['year'])
        return {'value': (award_wikidata['value']['award_series'], award_wikidata['value']['year'])}
