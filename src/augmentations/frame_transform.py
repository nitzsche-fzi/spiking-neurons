import numpy as np
import torch

class FrameTransform:
    """
    Transforms a sequence of events into frames.
    Designed to work with the tonic library.
    """
    def __init__(
            self, 
            dt, 
            n_steps, 
            original_sensor_size,
            desired_sensor_size, 
            random_time_scale = [1.0,1.0], #random scale factor for the time dimension
            random_image_scale = [1.0, 1.0], #random scale factor for the spatial dimensions
            random_image_offset = [0.0,0.0], #random offset for the spatial dimensions, in fractions of the sensor size
            random_start_offset=True,
            noise = 0.00, #probability for a random event to be generated per pixel per time step
            ):
        """
        Args:
        - dt: time step in milliseconds (e.g. 8.0)
        - n_steps: number of steps to slice the data into
        - original_sensor_size: the size of the original sensor (e.g. [128,128,2])
        - desired_sensor_size: the size of the desired sensor (e.g. [64,64,2])
        - random_time_scale: random scale factor for the time dimension (e.g. [0.9,1.1])
        - random_image_scale: random scale factor for the spatial dimensions (e.g. [0.9,1.1])
        - random_image_offset: random offset for the spatial dimensions, in fractions of the sensor size (e.g. [-0.1,0.1])
        - random_start_offset: random start offset for the time dimension, either True or False
        - noise: probability for a random event to be generated per pixel per time step (e.g. 0.01)

        """
        self.dt = dt #delta t per frame
        self.n_steps = n_steps #number of steps to slice the data into
        self.original_sensor_size = original_sensor_size
        self.desired_sensor_size = desired_sensor_size

        self.random_start_offset = random_start_offset
        self.random_time_scale = random_time_scale
        self.random_image_scale = random_image_scale
        self.random_image_offset = random_image_offset
        self.noise = noise  

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
        """
        rng = self._get_rng()

        dt, n_slices, start_time = self._compute_slices(events, rng)

        event_groups = self._get_event_groups(events, dt, n_slices, start_time)

        frames = self._create_frames(event_groups, rng) #rng needed for spatial scaling

        return frames

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
        frames = np.zeros((len(event_groups), *self.desired_sensor_size[::-1]), dtype=np.int16)
        image_scale = self._uniform(rng, *self.random_image_scale)
        image_offset = [self._uniform(rng, *self.random_image_offset), self._uniform(rng, *self.random_image_offset)]
        hx = self.original_sensor_size[0] / 2
        hy = self.original_sensor_size[1] / 2
        sx = self.desired_sensor_size[0] / self.original_sensor_size[0]
        sy = self.desired_sensor_size[1] / self.original_sensor_size[1]
        for i, event_slice in enumerate(event_groups):
            x = event_slice["x"].astype(float)
            y = event_slice["y"].astype(float)
            p = event_slice["p"]
            
            x = (x - hx) * image_scale + hx + image_offset[0] * self.original_sensor_size[0]
            y = (y - hy) * image_scale + hy + image_offset[1] * self.original_sensor_size[1]

            x = np.floor(x * sx).astype(int)
            y = np.floor(y * sy).astype(int)
            # drop events that are outside the sensor
            mask = (x >= 0) & (x < self.desired_sensor_size[0]) & (y >= 0) & (y < self.desired_sensor_size[1])
            x = x[mask]
            y = y[mask]
            p = p[mask].astype(int)

            np.add.at(
                frames,
                (i, p, y, x),
                1, 
            )
            # add noise to the frames
            noise = rng.random(frames[i].shape) < self.noise
            frames[i] += noise
        # frames[frames > 1] = 1 #binarize the data. fastest way to do it. 
        return frames
    
    def _compute_slices(self, events, rng):
        """Computes the local dt (with random time-scale) and determines 
        how many slices we can extract. Also chooses a random start time 
        if the stream is longer than needed."""
        
        times = events["t"]
        t_min = times[0]
        t_max = times[-1]

        # Apply random scaling to the nominal dt
        dt_local = self.dt * self._uniform(rng, *self.random_time_scale)

        # Naive number of slices
        naive_slices = int(np.floor(((t_max - t_min) - dt_local) / dt_local) + 1)
        
        # If user has asked for n_steps, adapt logic
        if self.n_steps > 0:
            # If naive_slices < self.n_steps, you might shrink dt or do your fallback:
            if naive_slices < self.n_steps:
                print("Warning: not enough data to create the desired number of steps.")
                # Example fallback: scale dt down to fit
                rate = naive_slices / (self.n_steps + 1)
                dt_local *= rate
                # Recalculate naive_slices
                naive_slices = int(np.floor(((t_max - t_min) - dt_local) / dt_local) + 1)
                # (You could also raise an error if you want.)

            # Use whichever is smaller, the naive_slices or the requested n_steps
            n_slices = min(naive_slices, self.n_steps)
        else:
            n_slices = naive_slices

        # Random sub-window if the data is longer than we need
        needed_window = dt_local * n_slices
        if self.random_start_offset and ((t_max - t_min) > needed_window):
            # Sample a random start time so that there's enough room for n_slices frames
            start_time = rng.uniform(t_min, t_max - needed_window)
        else:
            # If not enough data to shift around, just start at t_min
            start_time = t_min

        return dt_local, n_slices, start_time


    def _get_event_groups(self, events, dt_local, n_slices, start_time):
        """Slices the events into n_slices groups with a time window of dt_local each,
        starting from the float start_time."""
        times = events["t"]

        window_start_times = start_time + np.arange(n_slices) * dt_local
        window_end_times   = window_start_times + dt_local

        indices_start = np.searchsorted(times, window_start_times)
        indices_end   = np.searchsorted(times, window_end_times)

        event_groups = []
        for i in range(n_slices):
            event_groups.append(events[indices_start[i]:indices_end[i]])

        return event_groups

    