import random
from typing import Optional

import torch
from datasets import Dataset
from torch import Tensor
from torch_audiomentations.core.transforms_interface import (
    BaseWaveformTransform,
    EmptyPathException,
)
from torch_audiomentations.utils.dsp import calculate_rms
from torch_audiomentations.utils.object_dict import ObjectDict


class HFAddBackgroundNoise(BaseWaveformTransform):
    """
    Add background noise to the input audio.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    # Note: This transform has only partial support for multichannel audio. Noises that are not
    # mono get mixed down to mono before they are added to all channels in the input.
    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        dataset: Dataset,
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """

        :param background_paths: Either a path to a folder with audio files or a list of paths
            to audio files.
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximum SNR in dB.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """

        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        # TODO: check that one can read audio files
        self.dataset = dataset

        if len(dataset) == 0:
            raise EmptyPathException("There are no supported audio files found.")

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

    @staticmethod
    def rms_normalize(samples: Tensor) -> Tensor:
        """Power-normalize samples

        Parameters
        ----------
        samples : (..., time) Tensor
            Single (or multichannel) samples or batch of samples

        Returns
        -------
        samples: (..., time) Tensor
            Power-normalized samples
        """
        rms = samples.square().mean(dim=-1, keepdim=True).sqrt()
        return samples / (rms + 1e-8)

    def random_background(self, target_num_samples: int) -> torch.Tensor:
        pieces = []

        # TODO: support repeat short samples instead of concatenating from different files

        missing_num_samples = target_num_samples
        while missing_num_samples > 0:
            background_samples = self.dataset.select(random.randint(self.dataset))["audio"]
            background_num_samples = len(background_samples)

            if background_num_samples > missing_num_samples:
                sample_offset = random.randint(0, background_num_samples - missing_num_samples)
                num_samples = missing_num_samples
                background_samples = background_samples[sample_offset:num_samples]
                missing_num_samples = 0
            else:
                missing_num_samples -= background_num_samples

            pieces.append(background_samples)

        # the inner call to rms_normalize ensures concatenated pieces share the same RMS (1)
        # the outer call to rms_normalize ensures that the resulting background has an RMS of 1
        # (this simplifies "apply_transform" logic)
        return self.rms_normalize(torch.cat([self.rms_normalize(piece) for piece in pieces], dim=1))

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """

        :params samples: (batch_size, num_channels, num_samples)
        """

        batch_size, _, num_samples = samples.shape

        # (batch_size, num_samples) RMS-normalized background noise
        self.transform_parameters["background"] = torch.stack(
            [self.random_background(num_samples) for _ in range(batch_size)]
        )

        # (batch_size, ) SNRs
        if self.min_snr_in_db == self.max_snr_in_db:
            self.transform_parameters["snr_in_db"] = torch.full(
                size=(batch_size,),
                fill_value=self.min_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            )
        else:
            snr_distribution = torch.distributions.Uniform(
                low=torch.tensor(self.min_snr_in_db, dtype=torch.float32, device=samples.device),
                high=torch.tensor(self.max_snr_in_db, dtype=torch.float32, device=samples.device),
                validate_args=True,
            )
            self.transform_parameters["snr_in_db"] = snr_distribution.sample(sample_shape=(batch_size,))

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        # (batch_size, num_samples)
        background = self.transform_parameters["background"].to(samples.device)

        # (batch_size, num_channels)
        background_rms = calculate_rms(samples) / (
            10 ** (self.transform_parameters["snr_in_db"].unsqueeze(dim=-1) / 20)
        )

        background = background.view(batch_size, 1, num_samples).expand(-1, num_channels, -1)
        background = background_rms.unsqueeze(-1) * background

        return ObjectDict(
            samples=samples + background,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
