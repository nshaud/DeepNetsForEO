#!/usr/bin/env python
"""Command line utility to generate EO O&M metadata from SAFE files."""

import sys
import argparse

import s2reader


EOOM_TEMPLATE_PRODUCT = """<?xml version="1.0" encoding="UTF-8"?>
<opt:EarthObservation xmlns:opt="http://www.opengis.net/opt/2.1" xmlns:gml="http://www.opengis.net/gml/3.2"
  xmlns:eop="http://www.opengis.net/eop/2.1" xmlns:om="http://www.opengis.net/om/2.0"
  xmlns:ows="http://www.opengis.net/ows/2.0" xmlns:swe="http://www.opengis.net/swe/1.0" xmlns:wrs="http://www.opengis.net/cat/wrs/1.0"
  xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  gml:id="{eoIdentifier}" xsi:schemaLocation="http://www.opengis.net/opt/2.1 http://geo.spacebel.be/opensearch/xsd/opt.xsd"
>
  <om:phenomenonTime>
    <gml:TimePeriod gml:id="phenomenonTime_{eoIdentifier}">
      <gml:beginPosition>{timeStart}</gml:beginPosition>
      <gml:endPosition>{timeEnd}</gml:endPosition>
    </gml:TimePeriod>
  </om:phenomenonTime>
  <om:resultTime>
    <gml:TimeInstant gml:id="resultTime_{eoIdentifier}">
      <gml:timePosition>{availabilityTime}</gml:timePosition>
    </gml:TimeInstant>
  </om:resultTime>
  <om:procedure>
    <eop:EarthObservationEquipment gml:id="EarthObservationEquipment_{eoIdentifier}">
      <eop:platform>
        <eop:Platform>
          <eop:shortName>{eoPlatform}</eop:shortName>
          <eop:serialIdentifier>{eoPlatformSerialIdentifier}</eop:serialIdentifier>
          <eop:orbitType>{eoOrbitType}</eop:orbitType>
        </eop:Platform>
      </eop:platform>
      <eop:instrument>
        <eop:Instrument>
          <eop:shortName>{eoInstrument}</eop:shortName>
        </eop:Instrument>
      </eop:instrument>
      <eop:sensor>
        <eop:Sensor>
          <eop:sensorType>{eoSensorType}</eop:sensorType>
          <eop:operationalMode>{eoSensorMode}</eop:operationalMode>
          <eop:resolution uom="m">{eoResolution}</eop:resolution>
          <eop:swathIdentifier>{eoSwathIdentifier}</eop:swathIdentifier>
          <eop:wavelengthInformation>
            <eop:WavelengthInformation>
              <eop:discreteWavelengths uom="nm">{eoWavelengths}</eop:discreteWavelengths>
              <eop:spectralRange>{eoSpectralRange}</eop:spectralRange>
            </eop:WavelengthInformation>
          </eop:wavelengthInformation>
        </eop:Sensor>
      </eop:sensor>
      <eop:acquisitionParameters>
        <eop:Acquisition>
          <eop:orbitNumber>{eoOrbitNumber}</eop:orbitNumber>
          <eop:orbitDirection>{eoOrbitDirection}</eop:orbitDirection>
        </eop:Acquisition>
      </eop:acquisitionParameters>
    </eop:EarthObservationEquipment>
  </om:procedure>
  <om:observedProperty xlink:href="#phenom1" />
  <om:featureOfInterest>
    <eop:Footprint gml:id="FPN10020">
      <eop:multiExtentOf>
        <gml:MultiSurface gml:id="MultiSurface_{eoIdentifier}" srsName="http://www.opengis.net/def/crs/EPSG/0/4326">
          <gml:surfaceMembers>
            <gml:Polygon gml:id="Polygon_{eoIdentifier}" srsName="http://www.opengis.net/def/crs/EPSG/0/4326">
              <gml:exterior>
                <gml:LinearRing>
                  <gml:posList>{footprint}</gml:posList>
                </gml:LinearRing>
              </gml:exterior>
            </gml:Polygon>
          </gml:surfaceMembers>
        </gml:MultiSurface>
      </eop:multiExtentOf>
    </eop:Footprint>
  </om:featureOfInterest>
  <om:result>
    <opt:EarthObservationResult gml:id="EarthObservationResult_{eoIdentifier}">
      <opt:cloudCoverPercentage uom="%">{optCloudCover}</opt:cloudCoverPercentage>
      <!-- <opt:snowCoverPercentage uom="%">optSnowCover</opt:snowCoverPercentage> -->
    </opt:EarthObservationResult>
  </om:result>
  <eop:metaDataProperty>
    <eop:EarthObservationMetaData>
      <eop:identifier>{eoProductIdentifier}</eop:identifier>
      <eop:creationDate>{eoCreationDate}</eop:creationDate>
      <eop:modificationDate>{eoModificationDate}</eop:modificationDate>
      <eop:parentIdentifier>{eoParentIdentifier}</eop:parentIdentifier>
      <eop:acquisitionType>{eoAcquisitionType}</eop:acquisitionType>
      <eop:acquisitionSubType codeSpace="">{eoAcquisitionSubType}</eop:acquisitionSubType>
      <eop:productType>{eoProductType}</eop:productType>
      <eop:status>{eoProductionStatus}</eop:status>
      <eop:downlinkedTo>
        <eop:DownlinkInformation>
          <eop:acquisitionStation codeSpace="urn:eop:PHR:stationCode">{eoAcquisitionStation}</eop:acquisitionStation>
          <eop:acquisitionDate>{eoAcquisitionDate}</eop:acquisitionDate>
        </eop:DownlinkInformation>
      </eop:downlinkedTo>
      <eop:archivedIn>
        <eop:ArchivingInformation>
          <eop:archivingCenter codeSpace="urn:eop:PHR:stationCode">{archivingCenter}</eop:archivingCenter>
          <eop:archivingDate>{eoArchivingDate}</eop:archivingDate>
          <eop:archivingIdentifier codeSpace="urn:eop:PHR:stationCode">041028P600160013MC_00_4</eop:archivingIdentifier>
        </eop:ArchivingInformation>
      </eop:archivedIn>
      <eop:productQualityDegradation uom="">{eoProductQualityStatus}</eop:productQualityDegradation>
      <eop:productQualityDegradationTag codeSpace="">{eoProductQualityDegradationTag}</eop:productQualityDegradationTag>
      <eop:processing>
        <eop:ProcessingInformation>
          <eop:processingCenter>{eoProcessingCenter}</eop:processingCenter>
          <eop:processingDate>{eoProcessingDate}</eop:processingDate>
          <eop:compositeType>{eoCompositeType}</eop:compositeType>
          <eop:processorName>{eoProcessorName}</eop:processorName>
          <eop:processingLevel>{eoProcessingLevel}</eop:processingLevel>
          <eop:processingMode>{eoProcessingMode}</eop:processingMode>
        </eop:ProcessingInformation>
      </eop:processing>
    </eop:EarthObservationMetaData>
  </eop:metaDataProperty>
</opt:EarthObservation>"""

EOOM_TEMPLATE_GRANULE = """<?xml version="1.0" encoding="UTF-8"?>
<opt:EarthObservation xmlns:opt="http://www.opengis.net/opt/2.1" xmlns:gml="http://www.opengis.net/gml/3.2"
  xmlns:eop="http://www.opengis.net/eop/2.1" xmlns:om="http://www.opengis.net/om/2.0"
  xmlns:ows="http://www.opengis.net/ows/2.0" xmlns:swe="http://www.opengis.net/swe/1.0" xmlns:wrs="http://www.opengis.net/cat/wrs/1.0"
  xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  gml:id="{eoIdentifier}" xsi:schemaLocation="http://www.opengis.net/opt/2.1 http://geo.spacebel.be/opensearch/xsd/opt.xsd"
>
  <om:phenomenonTime>
    <gml:TimePeriod gml:id="phenomenonTime_{eoIdentifier}">
      <gml:beginPosition>{timeStart}</gml:beginPosition>
      <gml:endPosition>{timeEnd}</gml:endPosition>
    </gml:TimePeriod>
  </om:phenomenonTime>
  <om:resultTime>
    <gml:TimeInstant gml:id="resultTime_{eoIdentifier}">
      <gml:timePosition>{availabilityTime}</gml:timePosition>
    </gml:TimeInstant>
  </om:resultTime>
  <om:procedure>
    <eop:EarthObservationEquipment gml:id="EarthObservationEquipment_{eoIdentifier}">
      <eop:platform>
        <eop:Platform>
          <eop:shortName>{eoPlatform}</eop:shortName>
          <eop:serialIdentifier>{eoPlatformSerialIdentifier}</eop:serialIdentifier>
          <eop:orbitType>{eoOrbitType}</eop:orbitType>
        </eop:Platform>
      </eop:platform>
      <eop:instrument>
        <eop:Instrument>
          <eop:shortName>{eoInstrument}</eop:shortName>
        </eop:Instrument>
      </eop:instrument>
      <eop:sensor>
        <eop:Sensor>
          <eop:sensorType>{eoSensorType}</eop:sensorType>
          <eop:operationalMode>{eoSensorMode}</eop:operationalMode>
          <eop:resolution uom="m">{eoResolution}</eop:resolution>
          <eop:swathIdentifier>{eoSwathIdentifier}</eop:swathIdentifier>
          <eop:wavelengthInformation>
            <eop:WavelengthInformation>
              <eop:discreteWavelengths uom="nm">{eoWavelengths}</eop:discreteWavelengths>
              <eop:spectralRange>{eoSpectralRange}</eop:spectralRange>
            </eop:WavelengthInformation>
          </eop:wavelengthInformation>
        </eop:Sensor>
      </eop:sensor>
      <eop:acquisitionParameters>
        <eop:Acquisition>
          <eop:orbitNumber>{eoOrbitNumber}</eop:orbitNumber>
          <eop:orbitDirection>{eoOrbitDirection}</eop:orbitDirection>
          <!-- TODO: not supported by Sentinel-2?
          <eop:wrsLongitudeGrid codeSpace="">eoTrack</eop:wrsLongitudeGrid>
          <eop:wrsLatitudeGrid codeSpace="">eoFrame</eop:wrsLatitudeGrid>
          <eop:startTimeFromAscendingNode uom="s">eoStartTimeFromAscendingNode</eop:startTimeFromAscendingNode>
          <eop:completionTimeFromAscendingNode uom="s">eoCompletionTimeFromAscendingNode</eop:completionTimeFromAscendingNode>
          -->
          <eop:illuminationAzimuthAngle uom="deg">{eoIlluminationAzimuthAngle}</eop:illuminationAzimuthAngle>
          <eop:illuminationZenithAngle uom="deg">{eoIlluminationZenithAngle}</eop:illuminationZenithAngle>
          <!-- <eop:illuminationElevationAngle uom="deg">eoIlluminationElevationAngle</eop:illuminationElevationAngle> -->
        </eop:Acquisition>
      </eop:acquisitionParameters>
    </eop:EarthObservationEquipment>
  </om:procedure>
  <om:observedProperty xlink:href="#phenom1" />
  <om:featureOfInterest>
    <eop:Footprint gml:id="FPN10020">
      <eop:multiExtentOf>
        <gml:MultiSurface gml:id="MultiSurface_{eoIdentifier}" srsName="http://www.opengis.net/def/crs/EPSG/0/4326">
          <gml:surfaceMember>
            <gml:Polygon gml:id="Polygon_{eoIdentifier}" srsName="http://www.opengis.net/def/crs/EPSG/0/4326">
              <gml:exterior>
                <gml:LinearRing>
                  <gml:posList>{footprint}</gml:posList>
                </gml:LinearRing>
              </gml:exterior>
            </gml:Polygon>
          </gml:surfaceMember>
        </gml:MultiSurface>
      </eop:multiExtentOf>
    </eop:Footprint>
  </om:featureOfInterest>
  <om:result>
    <opt:EarthObservationResult gml:id="EarthObservationResult_{eoIdentifier}">
      <opt:cloudCoverPercentage uom="%">{optCloudCover}</opt:cloudCoverPercentage>
      <!-- <opt:snowCoverPercentage uom="%">optSnowCover</opt:snowCoverPercentage> -->
    </opt:EarthObservationResult>
  </om:result>
  <eop:metaDataProperty>
    <eop:EarthObservationMetaData>
      <eop:identifier>{eoProductIdentifier}</eop:identifier>
      <eop:creationDate>{eoCreationDate}</eop:creationDate>
      <eop:modificationDate>{eoModificationDate}</eop:modificationDate>
      <eop:parentIdentifier>{eoParentIdentifier}</eop:parentIdentifier>
      <eop:acquisitionType>{eoAcquisitionType}</eop:acquisitionType>
      <eop:acquisitionSubType codeSpace="">{eoAcquisitionSubType}</eop:acquisitionSubType>
      <eop:productType>{eoProductType}</eop:productType>
      <eop:status>{eoProductionStatus}</eop:status>
      <eop:downlinkedTo>
        <eop:DownlinkInformation>
          <eop:acquisitionStation codeSpace="urn:eop:PHR:stationCode">{eoAcquisitionStation}</eop:acquisitionStation>
          <eop:acquisitionDate>{eoAcquisitionDate}</eop:acquisitionDate>
        </eop:DownlinkInformation>
      </eop:downlinkedTo>
      <eop:archivedIn>
        <eop:ArchivingInformation>
          <eop:archivingCenter codeSpace="urn:eop:PHR:stationCode">{eoArchivingCenter}</eop:archivingCenter>
          <eop:archivingDate>{eoArchivingDate}</eop:archivingDate>
          <!--<eop:archivingIdentifier codeSpace="urn:eop:PHR:stationCode">041028P600160013MC_00_4</eop:archivingIdentifier>-->
        </eop:ArchivingInformation>
      </eop:archivedIn>
      <eop:productQualityDegradation uom="">{eoProductQualityStatus}</eop:productQualityDegradation>
      <eop:productQualityDegradationTag codeSpace="">{eoProductQualityDegradationTag}</eop:productQualityDegradationTag>
      <eop:processing>
        <eop:ProcessingInformation>
          <eop:processingCenter>{eoProcessingCenter}</eop:processingCenter>
          <eop:processingDate>{eoProcessingDate}</eop:processingDate>
          <eop:compositeType>{eoCompositeType}</eop:compositeType>
          <eop:processorName>{eoProcessorName}</eop:processorName>
          <eop:processingLevel>{eoProcessingLevel}</eop:processingLevel>
          <eop:processingMode>{eoProcessingMode}</eop:processingMode>
        </eop:ProcessingInformation>
      </eop:processing>
    </eop:EarthObservationMetaData>
  </eop:metaDataProperty>
</opt:EarthObservation>"""


# def main(args=sys.argv[1:]):
#     """Generate EO O&M XML metadata."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument("filename", type=str, nargs=1)
#     parser.add_argument("--granule-id", dest="granule_id", action="append",
#         help=(
#             "Optional. Specify a granule to export metadata from. Can be "
#             "specified multiple times."
#         )
#     )
#     parser.add_argument("--out-template", "-t", dest="out_template",
#         help=(
#             r"Specify a template to generate filenames. Use the Python string "
#             r"format syntax (). Possible template tags are: {granule_id}, "
#             r"{band_list}, {resolution}. "
#         )
#     )
#     parser.add_argument("--out-file", "-f", dest="out_files", action="append",
#         help=(
#             "Specify a single output file for the metadata. Must be passed once "
#             "for every granule present/selected."
#         )
#     )
#     parser.add_argument("--resolution", "-r", dest="resolution",
#         type=int, default=10,
#         help=(
#             "Only produce metadata for bands of this resolution (in meters). "
#             "Default is 10."
#         )
#     )

#     parsed = parser.parse_args(args)
#     safe_pkg = s2reader.open(parsed.filename[0])

#     granules = safe_pkg.granules

#     # when granules are passed, perform a validation and subset the whole list
#     # of granules
#     if parsed.granule_ids:
#         granule_dict = dict(
#             (granule.granule_identifier, granule) for granule in granules
#         )
#         available_ids = granule_dict.keys()

#         missing_ids = set(parsed.granule_ids) - set(available_ids)
#         if missing_ids:
#             raise Exception('Could not find granule%s: ' % (
#                 "s" if len(missing_ids) > 1 else "",
#                 ", ".join(missing_ids)
#             ))

#         granules = [
#             granule_dict[granule_id] for granule_id in parsed.granule_ids
#         ]

#     # when out-files are passed, check that the length is equal to the granules
#     # to process.
#     if parsed.out_files:
#         if len(granules) != len(parsed.out_files):
#             raise Exception(
#                 "Invalid number of out-files passed. Expected %d, got %d."
#                 % (len(granules) != len(parsed.out_files))
#             )
#         out_files = parsed.out_files

#     elif parsed.out_template:
#         # use the template to generate filenames
#         out_files = [
#             parsed.out_template.format(**dict(
#                 granule_id=granule.granule_identifier,
#                 resolution=parsed.resolution
#             ))
#             for granule in granules
#         ]

#     else:
#         # make a list of "empty filenames"
#         out_files = [None] * len(granules)

#     for granule, out_file in zip(granules, out_files):
#         params = _get_template_params(safe_pkg, granule, parsed.resolution)
#         xml_string = EOOM_TEMPLATE.format(**params)

#         if out_file is not None:
#             with open(out_file, "w") as f:
#                 f.write(xml_string)
#         else:
#             print(
#                 "Granule ID %s:\n\n%s\n\n"
#                 % (granule.granule_identifier, xml_string)
#             )
#             pass


def main(args=sys.argv[1:]):
    """Generate EO O&M XML metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs=1)
    parser.add_argument("--granule-id", dest="granule_id",
        help=(
            "Optional. Specify a granule to export metadata from."
        )
    )
    parser.add_argument("--single-granule", dest="single_granule",
        action="store_true", default=False,
        help=(
            "When only one granule is contained in the package, include product "
            "metadata from this one granule. Fails when more than one granule "
            "is contained."
        )
    )
    parser.add_argument("--out-file", "-f", dest="out_file",
        help=(
            "Specify an output file to write the metadata to. By default, the "
            "XML is printed on stdout."
        )
    )
    parser.add_argument("--resolution", "-r", dest="resolution", default="10",
        help=(
            "Only produce metadata for bands of this resolution (in meters). "
            "Default is 10."
        )
    )

    parsed = parser.parse_args(args)

    try:
        safe_pkg = s2reader.open(parsed.filename[0])
    except IOError, e:
        parser.error('Could not open SAFE package. Error was "%s"' % e)

    granules = safe_pkg.granules

    granule = None
    if parsed.granule_id:
        granule_dict = dict(
            (granule.granule_identifier, granule) for granule in granules
        )
        try:
            granule = granule_dict[parsed.granule_id]
        except KeyError:
            parser.error('No such granule %r' % parsed.granule_id)

    elif parsed.single_granule:
        if len(granules) > 1:
            parser.error('Package contains more than one granule.')

        granule = granules[0]

    params = _get_product_template_params(safe_pkg, parsed.resolution)

    if granule:
        params.update(_get_granule_template_params(granule, parsed.resolution))
        xml_string = EOOM_TEMPLATE_GRANULE.format(**params)
    else:
        xml_string = EOOM_TEMPLATE_PRODUCT.format(**params)

    if parsed.out_file:
        with open(parsed.out_file, "w") as f:
            f.write(xml_string)
    else:
        print(xml_string)


def _get_product_template_params(safe_pkg, resolution):
    metadata = safe_pkg._product_metadata

    wavelengths = " ".join([
        spectral_information.findtext("Wavelength/CENTRAL")
        for spectral_information in metadata.findall(".//Spectral_Information")
        if spectral_information.findtext("RESOLUTION") == str(resolution)
    ])

    band_names = "_".join([
        spectral_information.attrib["physicalBand"]
        for spectral_information in metadata.findall(".//Spectral_Information")
        if spectral_information.findtext("RESOLUTION") == str(resolution)
    ])

    identifier = metadata.findtext('.//PRODUCT_URI')
    footprint = metadata.findtext('.//Global_Footprint/EXT_POS_LIST').strip()

    return {
        'timeStart': safe_pkg.product_start_time,
        'timeEnd': safe_pkg.product_stop_time,
        'eoParentIdentifier': "S2_MSI_L1C",
        'eoAcquisitionType': "NOMINAL",
        'eoOrbitNumber': safe_pkg.sensing_orbit_number,
        'eoOrbitDirection': safe_pkg.sensing_orbit_direction,
        'optCloudCover': metadata.findtext(".//Cloud_Coverage_Assessment"),
        'eoCreationDate': safe_pkg.generation_time,
        'eoProcessingMode': "DATA_DRIVEN",

        "footprint": footprint,

        'eoIdentifier': identifier,
        'eoProductIdentifier': "%s_%s" % (identifier, resolution),

        'originalPackageType': "application/zip",
        'eoProcessingLevel': safe_pkg.processing_level,
        'eoSensorType': "OPTICAL",
        'eoOrbitType': "LEO",
        'eoProductType': safe_pkg.product_type,
        'eoInstrument': safe_pkg.product_type[2:5],
        'eoPlatform': safe_pkg.spacecraft_name[0:10],
        'eoPlatformSerialIdentifier': safe_pkg.spacecraft_name[10:11],

        'availabilityTime': safe_pkg.generation_time,

        'eoSensorMode': "",
        'eoResolution': resolution,
        'eoSwathIdentifier': "",  # TODO
        'eoWavelengths': wavelengths,
        'eoSpectralRange': "",  # TODO


        # TODO: find out correlation
        'eoModificationDate': "",
        'eoAcquisitionSubType': "",
        'eoProductionStatus': "",

        'eoAcquisitionStation': "",
        'eoAcquisitionDate': "",
        'eoArchivingDate': "",
        'eoProductQualityStatus': "",
        'eoProductQualityDegradationTag': "",
        'eoProcessingCenter': "",
        'eoProcessingDate': "",
        'eoCompositeType': "",
        'eoProcessorName': "",
    }


def _get_granule_template_params(granule, resolution):
    metadata = granule._metadata
    # footprint = metadata.findtext('.//Global_Footprint/EXT_POS_LIST').strip()

    return {
        'eoArchivingCenter': metadata.findtext('.//ARCHIVING_CENTRE'),
        # 'footprint': " ".join(
        #     "%f %f" % coord
        #     for coord in granule.footprint.exterior.coords
        # ),
        # "footprint": " ".join(_swapped(footprint.split())),
        'eoIdentifier': granule.granule_identifier,
        'availabilityTime': metadata.findtext('.//ARCHIVING_TIME'),
        'eoArchivingDate': metadata.findtext('.//ARCHIVING_TIME'),

        # there does not seem to be an equivalent for Sentinel 2
        # 'eoTrack': "",
        # 'eoFrame': "",
        # 'eoStartTimeFromAscendingNode': "",
        # 'eoStartTimeFromAscendingNode': "",

        'eoIlluminationAzimuthAngle': metadata.findtext('.//Mean_Sun_Angle/AZIMUTH_ANGLE'),
        'eoIlluminationZenithAngle': metadata.findtext('.//Mean_Sun_Angle/ZENITH_ANGLE'),
        # 'eoIlluminationElevationAngle': "",

        # not in MD
        # 'optSnowCover': "",

        'eoProductIdentifier': "%s_%s" % (
            granule.granule_identifier, resolution
        ),
    }


def _swapped(coords):
    ret = []
    for i in range(len(coords))[::2]:
        print i
        ret.append(coords[i + 1])
        ret.append(coords[i])

    return ret


if __name__ == "__main__":
    main(sys.argv[1:])
