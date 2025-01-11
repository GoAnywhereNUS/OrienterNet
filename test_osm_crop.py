import time
import matplotlib.pyplot as plt

from maploc.demo import Demo
from maploc.osm.viz import GeoPlotter
from maploc.osm.tiling import TileManager, OsmiumDataStore
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images

# Increasing the number of rotations increases the accuracy but requires more GPU memory.
# The highest accuracy is achieved with num_rotations=360
# but num_rotations=64~128 is often sufficient.
# To reduce the memory usage, we can reduce the tile size in the next cell.
demo = Demo(num_rotations=72, device="cuda")

# no EXIF data: provide a coarse location prior as address
#image_path = "assets/query_zurich_1.jpeg"
#prior_address = "ETH CAB Zurich"
# image_path = "assets/nus_i4.jpeg"
# prior_address = "3 Research Link Singapore"

# Try out these other queries!
image_path = "assets/query_vancouver_1.JPG"
prior_address = "Vancouver Waterfront Station"

# image_path = "assets/query_vancouver_2.JPG"
# image_path = "assets/query_vancouver_3.JPG"
# prior_address = None # here we load the location prior from the exif

image, camera, gravity, proj, bbox = demo.read_input_image(
    image_path,
    prior_address=prior_address,
    tile_size_meters=32,  # try 64, 256, etc.
)

# Query OpenStreetMap for this area
t1 = time.time()
datastore = OsmiumDataStore(proj, bbox + 100)
t2 = time.time()
tiler = TileManager.from_bbox(proj, bbox + 10, demo.config.data.pixel_per_meter, datastore=datastore)
t3 = time.time()
canvas = tiler.query(bbox)
t4 = time.time()

print(f"OSM query time: {t2-t1}, Crop time: {t3-t2}, Canvas time:{t4-t3}")

# Show the inputs to the model: image and raster map
map_viz = Colormap.apply(canvas.raster)
plot_images([image, map_viz], titles=["input image", "OpenStreetMap raster"])
plot_nodes(1, canvas.raster[2], fontsize=6, size=10)

from maploc.utils.viz_localization import (
    likelihood_overlay,
    plot_dense_rotations,
    add_circle_inset,
)
from maploc.utils.viz_2d import features_to_RGB

# Run the inference
for _ in range(10):
    start_time = time.time()
    uv, yaw, prob, neural_map, image_rectified = demo.localize(
        image, camera, canvas, gravity=gravity
    )
    print(time.time() - start_time)

# Visualize the predictions
overlay = likelihood_overlay(prob.numpy().max(-1), map_viz.mean(-1, keepdims=True))
(neural_map_rgb,) = features_to_RGB(neural_map.numpy())
plot_images([overlay, neural_map_rgb], titles=["prediction", "neural map"])
ax = plt.gcf().axes[0]
ax.scatter(*canvas.to_uv(bbox.center), s=5, c="red")
plot_dense_rotations(ax, prob, w=0.005, s=1 / 25)
add_circle_inset(ax, uv)
plt.savefig("notebook")

