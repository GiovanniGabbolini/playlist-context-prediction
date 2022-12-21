"""
Created on Wed Apr 22 2020

@author Name Redacted Surname Redacted
"""

import musicbrainzngs
import logging
import re


def album_release_year(release_group_musicbrainz_id) -> 'year':
    """Retrieves the year an album was released. It relies on the first-release-date attribute
    of a release group in musicbrainz

    Arguments:
        release_group_musicbrainz_id {str} --

    Returns:
        Year string
    """
    release_group = musicbrainzngs.get_release_group_by_id(release_group_musicbrainz_id['value'])['release-group']

    try:
        date = release_group['first-release-date']
    except KeyError:
        logging.getLogger('root.features').warning(
            f"Release-group {release_group_musicbrainz_id['value']} has not first-release-date attribute")
        return None

    if re.match(r"\d{4}-\d{2}-\d{2}", date):
        return_value = {'value': date.split('-')[0]}

    elif re.match(r"\d{4}-\d{2}", date):
        return_value = {'value': date.split('-')[0]}

    elif re.match(r"\d{4}", date):
        return_value = {'value': date}

    else:
        logging.getLogger('root.features').warning(
            f"Incorrect first release date format for {release_group_musicbrainz_id['value']}, got {date}")
        return_value = None

    return return_value
