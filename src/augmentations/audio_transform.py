import numpy as np
import torch

class AudioTransform:
    """
    A custom transformation for audio data with extended data augmentation. Designed to work with the tonic library.
    """
    def __init__(
            self, 
            dt, #time step in milliseconds 
            original_sensor_size, #[700,1,1]
            desired_sensor_size, #[350,1,1]
            random_time_scale = [1.0,1.0], #random scale factor for the time dimension
            squeeze_thresh = 5.0 #sum of the events in a frame below this threshold will be dropped and also after
            ):
        """ 
        Args:
        - dt: time step in milliseconds (e.g. 8.0)
        - original_sensor_size: the size of the original sensor (e.g. [700,1,1])
        - desired_sensor_size: the size of the desired sensor (e.g. [350,1,1])
        - random_time_scale: random scale factor for the time dimension (e.g. [1.0,1.0])
        - squeeze_thresh: to cut off silent start and end of the recording (e.g. 5.0): 
        sum of the events in a frame below this threshold will be dropped and also after


        """
        self.dt = dt * 1000.0

        self.original_sensor_size = original_sensor_size
        self.desired_sensor_size = desired_sensor_size
        self.random_time_scale = random_time_scale
        self.squeeze_thresh = squeeze_thresh

    def _uniform(self, rng, a, b, size=None):
        """
        returns a random float between a and b
        """
        if a == b:
            return a
        return a + (b - a) * rng.random(size)
        
    def _randint(self, rng, a, b, size=None):
        """
        returns a random integer between a and b, in [a, b)
        """
        if a == b:
            return a
        return rng.integers(a, b, size)

    def __call__(self, events):
        """
        Main function to transform the events into frames.

        Args:
        - events: the events to transform. Works with the tonic library.
        """
        rng = self._get_rng()

        time_factor = self._uniform(rng, *self.random_time_scale)

        event_groups = self._get_event_groups(events, rng, time_factor)

        frames = self._create_frames(event_groups, rng) #rng needed for spatial scaling

        frames = self._squeeze_frames(frames)

        return frames
    
    def _squeeze_frames(self, frames):
        event_sum = frames.reshape((frames.shape[0], -1)).sum(axis=-1)

        # Find indices of frames that have a sum >= squeeze_thresh
        indices = np.where(event_sum >= self.squeeze_thresh)[0]

        if len(indices) == 0:
            # If no frames meet the condition, return the original frames
            return frames

        # Get the start and end indices
        start = indices[0]
        end = indices[-1]

        # Return the frames from start to end (inclusive)
        return frames[start:end+1]

    def _get_rng(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:  # Check if in a worker process
            # Seed numpy RNG with a combination of worker seed and possibly other unique identifiers
            seed = worker_info.seed + worker_info.id + np.random.randint(0, 1000000)
            seed = seed % (2**32 - 1)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        return rng

    def _create_frames(self, event_groups, rng):
        n_steps = len(event_groups)

        frames = np.zeros((n_steps, *self.desired_sensor_size), dtype=np.int16) # int8 might be too small if there's a lot of events

        scale = self.desired_sensor_size[0] / self.original_sensor_size[0]

        for i, event_slice in enumerate(event_groups):
            x = event_slice["x"].astype(float)
            
            x = np.floor(x * scale).astype(int)
            # drop events that are outside the sensor
            np.add.at(
                frames,
                (i, x),
                1, 
            )
        return frames

    def _get_event_groups(self, events, rng, time_factor):
        """
        Slices the events into n_slices groups of events with a time window of dt per frame.
        """
        times = events["t"]
        sample_dt = self.dt * time_factor #this gives us the dt per bin 
        n_bins_for_sample = int((times[-1] - times[0]) / sample_dt)
        time_shift = self._uniform(rng, 0, sample_dt) #randomly shift the time window a tiny bit to augment further
        window_start_times = np.arange(n_bins_for_sample) * sample_dt + times[0] + time_shift
        window_end_times = window_start_times + sample_dt
        indices_start = np.searchsorted(times, window_start_times)
        indices_end = np.searchsorted(times, window_end_times)
        event_groups = [] #[events[indices_start[i] : indices_end[i]] for i in range(n_slices)]
        for i in range(n_bins_for_sample):
            event_groups.append(events[indices_start[i] : indices_end[i]])
        return event_groups
    

class AudioPad:
    """This is a custom collate function for a pytorch dataloader to load multiple event recordings
    at once. It's intended to be used in combination with sparse tensors. All tensor sizes are
    extended to the largest one in the batch, i.e. the longest recording.

    Example:
        >>> dataloader = torch.utils.data.DataLoader(dataset,
        >>>                                          batch_size=10,
        >>>                                          collate_fn=tonic.collation.PadTensors(),
        >>>                                          shuffle=True)

    The purpose of this custom padder is to add noise to the padded tensors, so that the networks cannot 
    rely on the length of the recordings to make predictions.
    """

    def __init__(self, batch_first: bool = True, noise = 0.0):
        self.batch_first = batch_first
        self.noise = noise

    def __call__(self, batch):
        samples_output = []
        targets_output = []

        max_length = max([sample.shape[0] for sample, target in batch])
        for sample, target in batch:
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            if sample.is_sparse:
                sample.sparse_resize_(
                    (max_length, *sample.shape[1:]),
                    sample.sparse_dim(),
                    sample.dense_dim(),
                )
            else:
                sample = torch.cat(
                    (
                        sample,
                        torch.zeros(
                            max_length - sample.shape[0],
                            *sample.shape[1:],
                            device=sample.device
                        ),
                    )
                )
                # add some binary noise to the padded tensor with prob self.noise
                noise_mask = torch.rand(sample.shape) < self.noise
                sample[noise_mask] += 1

            samples_output.append(sample)
            targets_output.append(target)
        
        samples_output = torch.stack(samples_output, 0 if self.batch_first else 1)
        if len(targets_output[0].shape) > 1:
            targets_output = torch.stack(targets_output, 0 if self.batch_first else -1) 
        else:
            targets_output = torch.tensor(targets_output, device=target.device)
        return (samples_output, targets_output)
    