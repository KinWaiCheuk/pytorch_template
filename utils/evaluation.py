import torch
import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from sklearn.metrics import precision_recall_fscore_support

def posterior2pianoroll(onsets, frames, onset_threshold=0.5, frame_threshold=0.5):
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1 # Make sure the activation is only 1 time-step

    onset_diff = onset_diff & (frames==1) # New condition such that both onset and frame on to get a note

    return onset_diff    
    
def extract_notes_wo_velocity(pianoroll):
    """
    Finds the note timings based on the onsets and frames information
    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """

    pitches = []
    intervals = []

    for nonzero in torch.nonzero(pianoroll, as_tuple=False):
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame

        # This while loop is looking for where does the note ends        
        while pianoroll[offset, pitch].item():
            offset += 1
            if offset == pianoroll.shape[0]:
                break

        # After knowing where does the note start and end, we can return the pitch information (and velocity)        
        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])

    return np.array(pitches), np.array(intervals) 


def transcription_accuracy(pred, y, metrics, hop_length, sampling_rate, min_midi):
    pred_roll = posterior2pianoroll(pred, pred)


    p, r, f, _ = precision_recall_fscore_support(y.flatten(),
                                                 pred_roll.flatten(),
                                                 average='binary')
    metrics['metric/frame/precision'] = p
    metrics['metric/frame/recall'] = r
    metrics['metric/frame/f1'] = f           

    # Extracting notes
    p_ref, i_ref = extract_notes_wo_velocity(y)
    p_est, i_est = extract_notes_wo_velocity(pred_roll)

    scaling = hop_length / sampling_rate

    # Converting time steps to seconds and midi number to frequency
    i_ref = (i_ref * scaling).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(min_midi + midi) for midi in p_ref])
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(min_midi + midi) for midi in p_est])          

    # Calcualte note-wise metrics
    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metrics['metric/note/precision'] = p
    metrics['metric/note/recall'] = r
    metrics['metric/note/f1'] = f
    metrics['metric/note/overlap'] = o     

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metrics['metric/note-with-offsets/precision'] = p
    metrics['metric/note-with-offsets/recall'] = r
    metrics['metric/note-with-offsets/f1'] = f
    metrics['metric/note-with-offsets/overlap'] = o