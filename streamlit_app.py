import streamlit as st
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely import wkt
from shapely.validation import make_valid
import folium
from streamlit_folium import st_folium
import json
import io
from typing import List, Tuple
import numpy as np

def ensure_polygon_new(geom):
    """Ensure geometry is a valid Polygon"""
    if geom is None or geom.is_empty:
        return None
    
    # Make geometry valid
    geom = make_valid(geom)
    
    if isinstance(geom, Polygon):
        return geom if geom.is_valid else None
    elif isinstance(geom, MultiPolygon):
        # Return the largest polygon from MultiPolygon
        return max(geom.geoms, key=lambda p: p.area) if geom.geoms else None
    else:
        # Try to convert other geometries to polygon
        try:
            return geom.convex_hull if hasattr(geom, 'convex_hull') else None
        except:
            return None

def detect_overlaps(gdf):
    """Detect overlapping polygons and return updated GeoDataFrame"""
    gdf = gdf.copy()
    
    # Ensure we have a unique_id column
    if 'unique_id' not in gdf.columns:
        gdf['unique_id'] = range(len(gdf))
    
    # Initialize overlapping_with column
    gdf['overlapping_with'] = ''
    
    # Build spatial index for efficient overlap detection
    sindex = gdf.sindex
    
    overlaps_found = 0
    
    for idx, row in gdf.iterrows():
        current_geom = row['geometry']
        current_id = str(row['unique_id'])
        overlapping_ids = []
        
        if current_geom is None or current_geom.is_empty:
            continue
            
        # Find potential overlaps using spatial index
        possible_matches_idx = list(sindex.intersection(current_geom.bounds))
        possible_matches = gdf.iloc[possible_matches_idx]
        
        for match_idx, match_row in possible_matches.iterrows():
            if match_idx == idx:  # Skip self
                continue
                
            match_geom = match_row['geometry']
            match_id = str(match_row['unique_id'])
            
            if match_geom is None or match_geom.is_empty:
                continue
                
            # Check for actual overlap
            try:
                if current_geom.intersects(match_geom):
                    intersection = current_geom.intersection(match_geom)
                    # Only consider significant overlaps (not just touching boundaries)
                    if intersection.area > 1e-10:
                        overlapping_ids.append(match_id)
                        overlaps_found += 1
            except Exception as e:
                st.warning(f"Error checking overlap between {current_id} and {match_id}: {e}")
                continue
        
        if overlapping_ids:
            gdf.loc[idx, 'overlapping_with'] = ','.join(overlapping_ids)
    
    return gdf, overlaps_found

def fix_minor_overlaps(gdf, progress_bar=None):
    """
    Fixed and improved version of the overlap fixing logic
    """
    try:
        gdf = gdf.copy()
        
        # Ensure geometry column exists
        if 'geometry' not in gdf.columns:
            if 'polygon_corrected' in gdf.columns:
                gdf["geometry"] = gdf["polygon_corrected"].apply(wkt.loads)
            else:
                st.error("No geometry or polygon_corrected column found!")
                return gdf
        
        # Keep track of fixed polygons to avoid infinite loops
        fixed_polygons = set()
        total_overlaps = len(gdf[gdf['overlapping_with'] != ''])
        current_progress = 0
        
        # Process each row and its overlapping partners
        for idx, row in gdf.iterrows():
            if progress_bar:
                current_progress += 1
                progress_bar.progress(current_progress / len(gdf))
            
            if not row.get("overlapping_with", ""):  # Skip if no overlaps
                continue

            # Get the current polygon
            current_poly = row["geometry"]
            current_id = str(row["unique_id"])
            
            # Skip if already processed or invalid geometry
            if current_id in fixed_polygons or current_poly is None or current_poly.is_empty:
                continue

            # Get all overlapping partners
            overlapping_ids = [id.strip() for id in str(row["overlapping_with"]).split(",")]
            
            original_area = current_poly.area
            
            for partner_id in overlapping_ids:
                if not partner_id:  # Skip empty partner IDs
                    continue
                    
                # Get the partner polygon
                partner_mask = gdf["unique_id"].astype(str) == partner_id
                partner_row = gdf[partner_mask]
                
                if partner_row.empty:
                    continue

                partner_poly = partner_row["geometry"].values[0]
                
                if partner_poly is None or partner_poly.is_empty:
                    continue

                # Calculate intersection
                try:
                    intersection = current_poly.intersection(partner_poly)
                except Exception as e:
                    st.warning(f"Error calculating intersection between {current_id} and {partner_id}: {e}")
                    continue

                if not intersection.is_empty and intersection.area > 1e-10:
                    # Fix the current polygon by removing the overlapping part
                    try:
                        new_poly = current_poly.difference(partner_poly)
                        
                        # Handle different geometry types
                        if new_poly.is_empty:
                            # If difference results in empty geometry, try buffering
                            new_poly = current_poly.buffer(-1e-6).difference(partner_poly.buffer(1e-6))
                        
                        if isinstance(new_poly, MultiPolygon) and new_poly.geoms:
                            # Keep the largest polygon from MultiPolygon
                            new_poly = max(new_poly.geoms, key=lambda p: p.area)
                        elif not isinstance(new_poly, Polygon):
                            # Try to convert to polygon
                            new_poly = ensure_polygon_new(new_poly)
                        
                        # Apply small negative buffer to ensure clean boundaries
                        if new_poly and not new_poly.is_empty:
                            new_poly = new_poly.buffer(-1e-8)
                            new_poly = ensure_polygon_new(new_poly)
                        
                        # Only update if the result is valid and not too small
                        if new_poly and new_poly.is_valid and new_poly.area > original_area * 0.1:
                            # Update the geometry in the GeoDataFrame
                            gdf.loc[gdf["unique_id"].astype(str) == current_id, "geometry"] = new_poly
                            current_poly = new_poly  # Update for next iteration
                            
                    except Exception as e:
                        st.warning(f"Error fixing overlap between {current_id} and {partner_id}: {e}")
                        continue
            
            # Mark this polygon as processed
            fixed_polygons.add(current_id)
        
        # Clear overlapping_with column for fixed polygons
        gdf['overlapping_with'] = ''
        
        return gdf
        
    except Exception as e:
        st.error(f"Error in fix_minor_overlaps: {e}")
        return gdf

def create_map(gdf, title="Map"):
    """Create a folium map from GeoDataFrame"""
    if gdf.empty:
        return None
    
    # Calculate map center
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add polygons to map
    for idx, row in gdf.iterrows():
        if row['geometry'] and not row['geometry'].is_empty:
            # Different colors for overlapping vs non-overlapping
            color = 'red' if row.get('overlapping_with', '') else 'blue'
            
            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda feature, color=color: {
                    'fillColor': color,
                    'color': color,
                    'weight': 2,
                    'fillOpacity': 0.3,
                }
            ).add_to(m)
    
    return m

def main():
    st.title("🗺️ GeoJSON Overlap Fixer")
    st.markdown("Upload a GeoJSON file to detect and fix polygon overlaps")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a GeoJSON file", type=['geojson', 'json'])
    
    if uploaded_file is not None:
        try:
            # Load the GeoJSON
            if uploaded_file.name.endswith('.geojson') or uploaded_file.name.endswith('.json'):
                gdf = gpd.read_file(uploaded_file)
            else:
                st.error("Please upload a valid GeoJSON file")
                return
            
            st.success(f"✅ Loaded {len(gdf)} polygons from {uploaded_file.name}")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Polygons", len(gdf))
            with col2:
                valid_geoms = gdf['geometry'].apply(lambda x: x is not None and x.is_valid).sum()
                st.metric("Valid Geometries", valid_geoms)
            with col3:
                total_area = gdf['geometry'].area.sum()
                st.metric("Total Area", f"{total_area:.2f}")
            
            # Show first few rows
            with st.expander("📋 View Data"):
                st.dataframe(gdf.head())
            
            # Detect overlaps
            st.subheader("🔍 Overlap Detection")
            
            if st.button("Detect Overlaps", type="primary"):
                with st.spinner("Detecting overlaps..."):
                    gdf_with_overlaps, overlap_count = detect_overlaps(gdf)
                    st.session_state['gdf_with_overlaps'] = gdf_with_overlaps
                    st.session_state['overlap_count'] = overlap_count
            
            if 'gdf_with_overlaps' in st.session_state:
                overlap_count = st.session_state['overlap_count']
                gdf_with_overlaps = st.session_state['gdf_with_overlaps']
                
                if overlap_count > 0:
                    st.warning(f"⚠️ Found {overlap_count} overlapping polygon pairs")
                    
                    # Show overlapping polygons
                    overlapping_polygons = gdf_with_overlaps[gdf_with_overlaps['overlapping_with'] != '']
                    if not overlapping_polygons.empty:
                        with st.expander(f"📋 View {len(overlapping_polygons)} Overlapping Polygons"):
                            st.dataframe(overlapping_polygons[['unique_id', 'overlapping_with']])
                    
                    # Show map before fixing
                    st.subheader("🗺️ Map Before Fixing (Red = Overlapping)")
                    map_before = create_map(gdf_with_overlaps, "Before Fixing")
                    if map_before:
                        st_folium(map_before, width=700, height=400)
                    
                    # Fix overlaps
                    st.subheader("🔧 Fix Overlaps")
                    if st.button("Fix All Overlaps", type="primary"):
                        with st.spinner("Fixing overlaps..."):
                            progress_bar = st.progress(0)
                            gdf_fixed = fix_minor_overlaps(gdf_with_overlaps, progress_bar)
                            st.session_state['gdf_fixed'] = gdf_fixed
                            progress_bar.progress(1.0)
                        
                        st.success("✅ Overlaps fixed!")
                        
                        # Re-detect overlaps to verify
                        gdf_verified, remaining_overlaps = detect_overlaps(gdf_fixed)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Overlaps Before", overlap_count)
                        with col2:
                            st.metric("Overlaps After", remaining_overlaps)
                        
                        if remaining_overlaps == 0:
                            st.success("🎉 All overlaps successfully removed!")
                        else:
                            st.warning(f"⚠️ {remaining_overlaps} overlaps remaining (may need manual review)")
                        
                        # Show map after fixing
                        st.subheader("🗺️ Map After Fixing")
                        map_after = create_map(gdf_verified, "After Fixing")
                        if map_after:
                            st_folium(map_after, width=700, height=400)
                        
                        # Download fixed GeoJSON
                        st.subheader("💾 Download Fixed Data")
                        
                        # Convert to GeoJSON string
                        geojson_str = gdf_fixed.to_json()
                        
                        st.download_button(
                            label="Download Fixed GeoJSON",
                            data=geojson_str,
                            file_name=f"fixed_{uploaded_file.name}",
                            mime="application/json"
                        )
                        
                        # Show statistics
                        with st.expander("📊 Statistics"):
                            original_area = gdf['geometry'].area.sum()
                            fixed_area = gdf_fixed['geometry'].area.sum()
                            area_change = ((fixed_area - original_area) / original_area) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Original Area", f"{original_area:.2f}")
                            with col2:
                                st.metric("Fixed Area", f"{fixed_area:.2f}")
                            with col3:
                                st.metric("Area Change", f"{area_change:.2f}%")
                
                else:
                    st.success("✅ No overlaps detected!")
                    
                    # Show map
                    st.subheader("🗺️ Map View")
                    map_view = create_map(gdf_with_overlaps, "No Overlaps")
                    if map_view:
                        st_folium(map_view, width=700, height=400)
        
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.error("Please ensure your file is a valid GeoJSON format")

if __name__ == "__main__":
    main()
