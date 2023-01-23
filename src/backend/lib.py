import random
import multiprocessing as mp
from typing import Iterator, Dict, List

import pandas as pd
from mongoengine import (
    EmbeddedDocument,
    IntField,
    StringField,
    EmbeddedDocumentField,
    FloatField,
    Document,
    ListField,
    ReferenceField,
    BooleanField,
    DictField
)

IMAGE_DIR_PATH = "data/images"
SESSION_DIR_PATH = "data/sessions"
NO_REGION = "<NO_REGION>"


class Point(EmbeddedDocument):
    x = IntField()
    y = IntField()

    meta = {'allow_inheritance': True}


class RegionOfInterest(EmbeddedDocument):
    name = StringField()
    top_left = EmbeddedDocumentField(Point)
    bottom_right = EmbeddedDocumentField(Point)

    def is_inside(self, point: Point) -> bool:
        x_in = (point.x >= self.top_left.x) & (point.x <= self.bottom_right.x)
        y_in = (point.y >= self.top_left.y) & (point.y <= self.bottom_right.y)
        return x_in & y_in


class EyeTrackerPoint(Point):
    duration = FloatField()


class User(EmbeddedDocument):
    name = StringField()


class LabeledImage(Document):
    image_path = StringField()
    width = IntField()
    height = IntField()
    regions = ListField(EmbeddedDocumentField(RegionOfInterest))


class EyeTrackerDevice(EmbeddedDocument):

    meta = {'allow_inheritance': True}

    def stream_points(self, image: LabeledImage) -> Iterator[EyeTrackerPoint]:
        pass


class DummyEyeTrackerDevice(EyeTrackerDevice):
    num_points_to_send = IntField(default=10)

    def stream_points(self, image: LabeledImage) -> Iterator[EyeTrackerPoint]:
        for _ in range(self.num_points_to_send):
            yield EyeTrackerPoint(
                x=random.randint(0, image.height),
                y=random.randint(0, image.width),
                duration=random.random() * 5
            )


class EyeTrackerSessionItem(EmbeddedDocument):
    user = EmbeddedDocumentField(User)
    image = ReferenceField(LabeledImage)
    sequence = ListField(EmbeddedDocumentField(EyeTrackerPoint))
    region_pattern = ListField(StringField())
    recorded = BooleanField(default=False)

    def record(self, device: EyeTrackerDevice):
        for et_point in device.stream_points(self.image):
            self.sequence.append(et_point)
        self._map_points_to_regions()
        self.recorded = True

    def get_region_duration_histogram(self) -> Dict[str, float]:
        self._assert_recorded()
        region_durations = map(self._region_duration_worker, self.image.regions)
        return {
            region.name: duration
            for region, duration in zip(self.image.regions, region_durations)
        }

    def _assert_recorded(self):
        assert self.recorded, "Session needs to be recorded first."

    def _map_points_to_regions(self):
        self.region_pattern = list(map(
            self._point_to_region_worker, self.sequence
        ))

    def _point_to_region_worker(self, point: EyeTrackerPoint) -> int:
        for region in self.image.regions:
            if region.is_inside(point):
                return region.name
        return NO_REGION

    def _region_duration_worker(self, region: RegionOfInterest) -> float:
        total_duration: float = 0.0
        for et_point, mapped_region in zip(self.sequence, self.region_pattern):
            if mapped_region == region.name:
                total_duration += et_point.duration
        return total_duration


class EyeTrackerSession(Document):
    image = ReferenceField(LabeledImage)
    device = EmbeddedDocumentField(EyeTrackerDevice)
    user_sessions = DictField()

    def record_single_session(self, user: User) -> EyeTrackerSessionItem:
        user_session = EyeTrackerSessionItem(user=user, image=self.image)
        user_session.record(device=self.device)
        self.user_sessions[user.name] = user_session
        return user_session

    def get_session_for_user(self, user_name: str) -> EyeTrackerSessionItem:
        if user_name not in self.user_sessions:
            raise RuntimeError(f"No session has been recorded for user: {user_name}")
        return self.user_sessions[user_name]

    def export_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict({
            "id": self.user_sessions.keys(),
            "sequence": [
                user_session_item.region_pattern
                for user_session_item in self.user_sessions.values()
            ]
        })
