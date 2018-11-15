#!/usr/bin/env python
"""Command line utility to inspect SAFE files."""

import sys
import argparse
import s2reader
import pprint


def main(args=None):
    """Print metadata as JSON strings."""
    args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("safe_file", type=str, nargs='+')
    parser.add_argument("--granules", action="store_true")
    parsed = parser.parse_args(args)

    pp = pprint.PrettyPrinter()
    for safe_file in parsed.safe_file:
        with s2reader.open(safe_file) as safe_dataset:
            if parsed.granules:
                pp.pprint(
                    dict(
                        safe_file=safe_file,
                        granules=[
                            dict(
                                granule_identifier=granule.granule_identifier,
                                footprint=str(granule.footprint),
                                srid=granule.srid,
                                # cloudmask_polys=str(granule.cloudmask),
                                # nodata_mask=str(granule.nodata_mask),
                                cloud_percent=granule.cloud_percent
                                )
                            for granule in safe_dataset.granules
                            ]
                        )
                    )
            else:
                pp.pprint(
                    dict(
                        safe_file=safe_file,
                        product_start_time=safe_dataset.product_start_time,
                        product_stop_time=safe_dataset.product_stop_time,
                        generation_time=safe_dataset.generation_time,
                        footprint=str(safe_dataset.footprint),
                        bounds=str(safe_dataset.footprint.bounds),
                        granules=len(safe_dataset.granules),
                        granules_srids=list(set([
                            granule.srid
                            for granule in safe_dataset.granules
                            ]))
                        )
                    )
            print "\n"


if __name__ == "__main__":
    main()
