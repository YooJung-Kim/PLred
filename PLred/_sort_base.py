"""
Instrument-agnostic frame sorting utilities for PLred.

This module contains generic functions for timestamp matching and frame binning
that are reused across different instruments (visPLred, IRPLred, etc).

Extracted from sort.py to promote code reuse and reduce duplication.
"""

import numpy as np
import glob
import re
import os
from datetime import datetime
from bisect import bisect


def find_data_between(datadir, obs_start, obs_end, header='', footer=''):
    """
    Find data files between specified observation times.

    The filename is expected to contain a timestamp in HH:MM:SS.ffffff format.

    Parameters
    ----------
    datadir : str
        Path to directory containing data files
    obs_start : str
        Start time in format %H:%M:%S (e.g., '11:59:00')
    obs_end : str
        End time in format %H:%M:%S (e.g., '12:20:09')
    header : str, optional
        Prefix pattern for files to match
    footer : str, optional
        Suffix pattern for files to match

    Returns
    -------
    valid_files : list
        Sorted list of files with timestamps between obs_start and obs_end
    """
    start = datetime.strptime(obs_start, "%H:%M:%S")
    end = datetime.strptime(obs_end, "%H:%M:%S")

    files = glob.glob(datadir + header + '*' + footer)
    files = sorted(files)

    pattern = r"(\d{2}:\d{2}:\d{2}\.\d+)"

    valid_files = []

    for f in files:
        match = re.search(pattern, f)

        if match:
            obstime = match.group(1)
            obstime = datetime.strptime(obstime[:13], "%H:%M:%S.%f")

            if (obstime > start) and (obstime < end):
                valid_files.append(f)

    print("number of files found: %d" % len(valid_files))

    return valid_files


def validate_timestamp_matching(timestamps1, timestamps2):
    """
    Validate that timestamps from two data streams match.

    Uses a two-pointer scan to find exact matches between sorted timestamp arrays.

    Parameters
    ----------
    timestamps1 : ndarray
        First timestamp array (unix epoch)
    timestamps2 : ndarray
        Second timestamp array (unix epoch)

    Returns
    -------
    idx1 : ndarray of bool
        Boolean array indicating which timestamps1 entries matched
    idx2 : ndarray of bool
        Boolean array indicating which timestamps2 entries matched
    """
    print(
        "Timestamp1 start: %s, end %s, length %d "
        % (
            datetime.fromtimestamp(timestamps1[0]),
            datetime.fromtimestamp(timestamps1[-1]),
            len(timestamps1),
        )
    )
    print(
        "Timestamp2 start: %s, end %s, length %d "
        % (
            datetime.fromtimestamp(timestamps2[0]),
            datetime.fromtimestamp(timestamps2[-1]),
            len(timestamps2),
        )
    )

    idx1 = [False] * len(timestamps1)
    idx2 = [False] * len(timestamps2)

    i, j = 0, 0
    while i < len(timestamps1) and j < len(timestamps2):
        if timestamps1[i] == timestamps2[j]:
            idx1[i] = True
            idx2[j] = True
            i += 1
            j += 1
        elif timestamps1[i] < timestamps2[j]:
            i += 1
        else:
            j += 1

    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    print(
        "Filtered %d out of timestamp1, %d out of timestamp2"
        % (np.sum(~(idx1)), np.sum(~(idx2)))
    )

    return idx1, idx2


def bin_by_centroids_from_indices(psfcamframes, centroids, xbins, ybins):
    """
    Bin PSF camera frames by centroid locations into spatial grid bins.

    Given PSF frame centroids and spatial bin edges, averages PSF frames
    within each bin and returns bin metadata for later frame lookup.

    Parameters
    ----------
    psfcamframes : ndarray, shape (N_frames, H, W)
        PSF camera frames to bin
    centroids : ndarray, shape (N_frames, 2)
        (x, y) centroid positions for each frame in pixels
    xbins : ndarray
        X-axis bin edges
    ybins : ndarray
        Y-axis bin edges

    Returns
    -------
    psfcam_binned_frames : ndarray, shape (N_xbins-1, N_ybins-1, H, W)
        Average PSF frame for each spatial bin
    num_frames : ndarray, shape (N_xbins-1, N_ybins-1)
        Number of frames contributing to each bin
    idxs : ndarray, shape (N_xbins-1, N_ybins-1, N_frames), dtype=bool
        Boolean mask indicating which frames fall into each bin
    """
    x = centroids[:, 0]
    y = centroids[:, 1]

    psfcam_binned_frames = np.zeros(
        (len(xbins) - 1, len(ybins) - 1, psfcamframes.shape[1], psfcamframes.shape[2])
    )
    num_frames = np.zeros((len(xbins) - 1, len(ybins) - 1))
    idxs = np.zeros((len(xbins) - 1, len(ybins) - 1, len(centroids)), dtype=bool)

    for i in range(len(xbins) - 1):
        for j in range(len(ybins) - 1):
            # Check if x and y coordinates are within the bin
            xidx = (x >= xbins[i]) & (x < xbins[i + 1])
            yidx = (y >= ybins[j]) & (y < ybins[j + 1])
            idx = xidx & yidx

            # Store the indices of frames that match this bin
            idxs[i, j] = idx
            if np.sum(idx) > 0:
                psfcam_binned_frames[i, j] = np.mean(psfcamframes[idx], axis=0)
            num_frames[i, j] = np.sum(idx)

    return psfcam_binned_frames, num_frames, idxs
