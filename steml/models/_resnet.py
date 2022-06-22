from typing import Tuple, List
from tensorflow import Tensor
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class ResidualBlock:
    def __init__(self, filters: int, kernel_size: int, new_stack: bool) -> None:
        self.proj1 = None
        if new_stack:
            self.proj1 = Conv2D(
                filters=filters,
                kernel_size=1,
                strides=2,
                padding='same',
            )
        self.conv1 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=2 if new_stack else 1,
            padding='same',
        )
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv2 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
        )
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()

    def __call__(self, x: Tensor) -> Tensor:
        residual = x
        if self.proj1 is not None:
            residual = self.proj1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu2(x)
        return x


class ResidualStack:
    def __init__(self, filters: int, kernel_size: int, blocks: int, new_stack: bool = True) -> None:
        self.blocks = [
            ResidualBlock(filters=filters, kernel_size=kernel_size, new_stack=(i == 0) & new_stack)
            for i in range(blocks)
        ]

    def __call__(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ResNet18(Model):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        activation: str,
        lr: float,
        loss: str,
        metrics: List[str],
    ) -> None:
        conv1 = Conv2D(filters=64, kernel_size=7, strides=2, padding='same')
        max_pool = MaxPooling2D(pool_size=3, strides=2, padding='same')
        conv2_stack = ResidualStack(filters=64, kernel_size=3, blocks=2, new_stack=False)
        conv3_stack = ResidualStack(filters=128, kernel_size=3, blocks=2)
        conv4_stack = ResidualStack(filters=256, kernel_size=3, blocks=2)
        conv5_stack = ResidualStack(filters=512, kernel_size=3, blocks=2)
        avg_pool = GlobalAveragePooling2D()
        fc1 = Dense(units=num_classes, activation=activation)

        inputs = Input(shape=input_shape)
        x = inputs
        x = conv1(x)
        x = max_pool(x)
        x = conv2_stack(x)
        x = conv3_stack(x)
        x = conv4_stack(x)
        x = conv5_stack(x)
        x = avg_pool(x)
        x = fc1(x)
        outputs = x
        super().__init__(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=lr)

        self.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        self.summary()
