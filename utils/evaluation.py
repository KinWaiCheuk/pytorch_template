import torch
import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from sklearn.metrics import precision_recall_fscore_support

# def posterior2pianoroll(onsets, frames, onset_threshold=0.5, frame_threshold=0.5):
#     onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
#     frames = (frames > frame_threshold).cpu().to(torch.uint8)
#     onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1 # Make sure the activation is only 1 time-step

# #     onset_diff = onset_diff & (frames==1) # New condition such that both onset and frame on to get a note
#     return frames, onset_diff
#     return onset_diff

    
def extract_notes_wo_velocity(frames, onsets):
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
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1 # Make sure the activation is only 1 time-step
    onset_diff = onset_diff & (frames==1) # New condition such that both onset and frame on to get a note
    
    pitches = []
    intervals = []

    # find the non-zero indices for onset_diff
    # the non-zero indices are the time (frame) and pitch information
    nonzero_tensor = torch.nonzero(onset_diff, as_tuple=False)

    for nonzero in nonzero_tensor:
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame

        # This while loop is looking for where does the note ends
        while frames[offset, pitch].item():
            offset += 1
            if offset == frames.shape[0]:
                break

        # After knowing where does the note start and end, we can return the pitch information (and velocity)        
        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])

    return np.array(pitches), np.array(intervals) 


class Evaluator():
    """
    A class for evaluating transcription accuracy.
    Current version only support one sample of the shape (T, F)
    Example usage:
    evaluator = Evaluator(hop_length, sr, min_midi)
    metrics = evaluator.evaluate(pred_frame, pred_onset, y_frame, y_onset)
    
    Parameters
    ----------
    hop_length: int
        Hop length in samples
    sr: int
        Sampling rate
    min_midi: int
        Minimum midi number
    onset_threshold: float
        Threshold for onset detection
    frame_threshold: float
        Threshold for frame detection
    """

    def __init__(
        self,
        hop_length,
        sampling_rate,
        min_midi,
        onset_threshold=0.5,
        frame_threshold=0.5            
    ):
        
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.min_midi = min_midi
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold


    def evaluate(
        self,
        pred_frame,
        pred_onset,
        y_frame,
        y_onset
    ):
        
        """
        A method for evaluating transcription accuracy.
        Current version only support one sample of the shape (T, F).
        T = frames
        F = bins
        
        Parameters
        ----------
        pred_frame: torch.FloatTensor, shape = [frames, bins]
        pred_onset: torch.FloatTensor, shape = [frames, bins]
        y_frame: torch.FloatTensor, shape = [frames, bins]
        y_onset: torch.FloatTensor, shape = [frames, bins]

        Returns
        -------
        metrics: dict  
            A dictionary of different metrics such as frame, note, and note with offsets.
            Each metric consists of (p=precision, r=recall, f=f1).
        """        
    
        metrics = {}
        
        pred_frame = (pred_frame > self.frame_threshold).to(torch.uint8)
        pred_onset = (pred_onset > self.onset_threshold).to(torch.uint8)


        p, r, f, _ = precision_recall_fscore_support(y_frame.flatten().cpu(),
                                                    pred_frame.flatten().cpu(),
                                                    average='binary')
        metrics['frame/precision'] = p
        metrics['frame/recall'] = r
        metrics['frame/f1'] = f           

        # Extracting notes
        p_ref, i_ref = extract_notes_wo_velocity(y_frame, y_onset)
        p_est, i_est = extract_notes_wo_velocity(pred_frame, pred_onset)

        scaling = self.hop_length / self.sampling_rate

        # Converting time steps to seconds and midi number to frequency
        i_ref = (i_ref * scaling).reshape(-1, 2)
        p_ref = np.array([midi_to_hz(self.min_midi + midi) for midi in p_ref])
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(self.min_midi + midi) for midi in p_est])          

        # Calcualte note-wise metrics
        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
        metrics['note/precision'] = p
        metrics['note/recall'] = r
        metrics['note/f1'] = f

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        metrics['note-with-offsets/precision'] = p
        metrics['note-with-offsets/recall'] = r
        metrics['note-with-offsets/f1'] = f
        
        return metrics