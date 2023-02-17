import astropy.time

from dataclasses import dataclass
from .Target import Target


@dataclass
class Orbit():

    
    orbit_id: str
    time_range: tuple[astropy.time.Time]
    targets: tuple[Target]


    @property
    def start(self) -> astropy.time.Time:
        return self.time_range[0]
    

    @property
    def end(self) -> astropy.time.Time:
        return self.time_range[1]


    def _register_targets(self):

        for i, target in enumerate(self.targets):
            b_after_start = target.start >= self.start
            b_before_end = target.start <= self.end
            if b_after_start and b_before_end:
                target.orbit = self
            else:
                raise ValueError(f'Target \'{target.target_id}\' not within ' +
                    f'orbit \'{self.orbit_id}\' time range')

        self.sort_targets()


    def sort_targets(self):

        # Sort targets by time: https://stackoverflow.com/a/613218
        dt_times = {i: t.start.datetime for i, t in enumerate(self.targets)}
        s = {k: v for k, v in sorted(dt_times.items(), key=lambda item: item[1])}
        self.targets = [self.targets[i] for i in s.keys()]


    def plot_targets(self, out_dir: str = None) -> tuple:

        self._register_targets()
        for i, target in enumerate(self.targets, start=1):
            fig, axs = target.plot(out_dir)

        return fig, axs