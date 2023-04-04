class Options:
    def __init__(
        self, focal=1015.0, center=112.0, z_near=5.0, z_far=15.0, camera_d=10.0
    ) -> None:
        self.focal = focal
        self.center = center
        self.z_near = z_near
        self.z_far = z_far
        self.camera_distance = camera_d
