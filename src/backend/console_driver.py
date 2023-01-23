import random
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mongoengine import connect
from sgt import SGT
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from lib import (
    EyeTrackerSession,
    User,
    LabeledImage,
    RegionOfInterest,
    Point,
    DummyEyeTrackerDevice
)

connect(db='sotis-db', host='127.0.0.1', port=27017)


def main():
    image = LabeledImage(
        image_path="some path to image",
        width=100,
        height=100,
        regions=[
            RegionOfInterest(name="region0", top_left=Point(x=1, y=1), bottom_right=Point(x=5, y=5)),
            RegionOfInterest(name="region1", top_left=Point(x=8, y=8), bottom_right=Point(x=20, y=20)),
            RegionOfInterest(name="region2", top_left=Point(x=40, y=40), bottom_right=Point(x=80, y=80))
        ]
    )
    image.save()
    device = DummyEyeTrackerDevice(num_points_to_send=5)
    users = [
        User(name=''.join(random.choices(string.ascii_uppercase, k=5)))
        for _ in range(30)
    ]
    session = EyeTrackerSession(image=image, device=device)
    for user_idx, user in enumerate(users):
        session.record_single_session(user=user)
        print(f"Finished recording a session {user_idx + 1} / {len(users)}")

    user_session = session.get_session_for_user(users[0].name)
    print(user_session.sequence)
    print(user_session.region_pattern)
    print(user_session.get_region_duration_histogram())
    print(user_session.region_pattern)
    sgt_ = SGT(kappa=1,
               lengthsensitive=False,
               mode='multiprocessing')
    sgtembedding_df = sgt_.fit_transform(session.export_df())
    sgtembedding_df = sgtembedding_df.set_index('id')
    print(sgtembedding_df)
    pca = PCA(n_components=2)
    pca.fit(sgtembedding_df)

    X = pca.transform(sgtembedding_df)

    print(np.sum(pca.explained_variance_ratio_))
    df = pd.DataFrame(data=X, columns=['x1', 'x2'])
    print(df.head())
    kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300)
    kmeans.fit(df)

    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_

    colmap = {1: 'r', 2: 'g', 3: 'b'}
    colors = list(map(lambda x: colmap[x + 1], labels))
    plt.scatter(df['x1'], df['x2'], color=colors, alpha=0.5, edgecolor=colors)
    plt.show()
    session.save()


if __name__ == "__main__":
    main()
