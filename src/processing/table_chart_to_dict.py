#!/usr/bin/env python3
"""
Table and Chart to Dictionary Converter

Converts extracted tables and charts into clean key-value pair dictionaries
for easy querying and data access.
"""

import json
from typing import Dict, List, Any
from pathlib import Path


class TableChartDictConverter:
    """Convert tables and charts to structured dictionaries."""

    def __init__(self, extraction_file: str):
        """
        Initialize with extraction results file.
        
        Args:
            extraction_file: Path to figures_tables JSON file
        """
        with open(extraction_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tables = self.data.get('tables', [])
        self.figures = self.data.get('figures', [])
        
        print(f"✓ Loaded {len(self.tables)} tables and {len(self.figures)} figures")

    def table_to_dict(self, table: Dict) -> Dict:
        """
        Convert a table to a clean dictionary structure.
        
        Strategy:
        - If table has clear row labels, use them as keys
        - If table is a comparison table, create nested structure
        - Preserve units and footnotes
        """
        result = {
            'metadata': {
                'title': table.get('title', 'Untitled'),
                'type': table.get('type', 'unknown'),
                'page': table.get('page', 0),
                'table_number': table.get('table_number', 'Unknown')
            },
            'data': {},
            'units': table.get('units', {}),
            'footnotes': table.get('footnotes', []),
            'sample_size': table.get('sample_size', 'Not specified')
        }
        
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        if not headers or not rows:
            return result
        
        # Strategy 1: First column is row labels (most common)
        if len(headers) > 1:
            for row in rows:
                if len(row) > 0:
                    row_label = row[0]
                    row_data = {}
                    
                    # Map remaining columns to headers
                    for i, value in enumerate(row[1:], start=1):
                        if i < len(headers):
                            column_name = headers[i]
                            row_data[column_name] = value
                    
                    result['data'][row_label] = row_data
        
        # Strategy 2: Single column table (list of values)
        elif len(headers) == 1:
            result['data'][headers[0]] = [row[0] for row in rows if row]
        
        return result

    def table_to_flat_dict(self, table: Dict) -> Dict:
        """
        Convert table to completely flat key-value pairs.
        
        Format: "Table_Number.Row_Label.Column_Name" = value
        """
        flat = {}
        
        # Use table number if available, otherwise use title
        table_number = table.get('table_number', '')
        table_title = table.get('title', 'Untitled')
        
        if table_number and table_number != 'Unknown':
            prefix = table_number.replace(' ', '_').replace('.', '_')
        else:
            prefix = table_title.replace(' ', '_').replace('.', '_')[:50]
        
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        for row in rows:
            if len(row) > 0:
                row_label = row[0].replace(' ', '_').replace('.', '_')
                
                for i, value in enumerate(row[1:], start=1):
                    if i < len(headers):
                        column_name = headers[i].replace(' ', '_').replace('.', '_')
                        key = f"{prefix}.{row_label}.{column_name}"
                        flat[key] = value
        
        # Add metadata
        flat[f"{prefix}.page"] = table.get('page', 0)
        flat[f"{prefix}.type"] = table.get('type', 'unknown')
        flat[f"{prefix}.title"] = table_title
        
        return flat

    def survival_curve_to_dict(self, figure: Dict) -> Dict:
        """
        Convert Kaplan-Meier survival curve to structured dictionary.
        
        Most important data for clinical trials.
        """
        result = {
            'metadata': {
                'title': figure.get('title', 'Untitled'),
                'type': figure.get('type', 'unknown'),
                'page': figure.get('page', 0),
                'figure_number': figure.get('figure_number', 'Unknown')
            },
            'statistical_results': {
                'p_value': figure.get('p_value', 'Not specified'),
                'log_rank_test': figure.get('log_rank_test', 'Not specified'),
                'hazard_ratio': figure.get('hazard_ratio', 'Not specified')
            },
            'treatment_groups': {}
        }
        
        # Extract data for each treatment group
        groups = figure.get('groups', [])
        for group in groups:
            group_name = group.get('name', 'Unknown')
            
            result['treatment_groups'][group_name] = {
                'median_survival_months': group.get('median_survival_months', 'Not specified'),
                'median_survival_ci': group.get('median_survival_confidence_interval', 'Not specified'),
                'survival_rates': group.get('survival_rates', {}),
                'number_at_risk': group.get('number_at_risk', {})
            }
        
        return result

    def chart_to_dict(self, figure: Dict) -> Dict:
        """
        Convert any chart/figure to dictionary.
        
        Handles bar charts, line plots, etc.
        """
        result = {
            'metadata': {
                'title': figure.get('title', 'Untitled'),
                'type': figure.get('type', 'unknown'),
                'page': figure.get('page', 0),
                'figure_number': figure.get('figure_number', 'Unknown'),
                'caption': figure.get('caption', '')
            },
            'axes': {
                'x_axis': figure.get('x_axis', {}),
                'y_axis': figure.get('y_axis', {})
            },
            'data': figure.get('extracted_values', {}),
            'groups': figure.get('groups', [])
        }
        
        return result

    def convert_all_tables(self) -> Dict[str, Dict]:
        """Convert all tables to dictionaries using actual table names."""
        converted = {}
        
        for i, table in enumerate(self.tables):
            # Use actual table title/number from document
            table_number = table.get('table_number', f'Table {i+1}')
            table_title = table.get('title', 'Untitled')
            
            # Create clean key from table number and title
            if table_number and table_number != 'Unknown':
                # Use table number as primary key
                table_key = table_number.replace(' ', '_').replace('.', '_')
            else:
                # Fall back to sanitized title
                table_key = table_title.replace(' ', '_').replace('.', '_')[:50]
            
            # Ensure uniqueness
            if table_key in converted:
                table_key = f"{table_key}_{i+1}"
            
            converted[table_key] = self.table_to_dict(table)
        
        return converted

    def convert_all_figures(self) -> Dict[str, Dict]:
        """Convert all figures to dictionaries using actual figure names."""
        converted = {}
        
        for i, figure in enumerate(self.figures):
            # Use actual figure number and title from document
            figure_number = figure.get('figure_number', f'Figure {i+1}')
            figure_title = figure.get('title', 'Untitled')
            
            # Create clean key from figure number and title
            if figure_number and figure_number != 'Unknown':
                # Use figure number as primary key
                figure_key = figure_number.replace(' ', '_').replace('.', '_')
            else:
                # Fall back to sanitized title
                figure_key = figure_title.replace(' ', '_').replace('.', '_')[:50]
            
            # Ensure uniqueness
            if figure_key in converted:
                figure_key = f"{figure_key}_{i+1}"
            
            # Use specialized converter for survival curves
            figure_type = figure.get('type', 'unknown')
            if figure_type == 'survival_curve':
                converted[figure_key] = self.survival_curve_to_dict(figure)
            else:
                converted[figure_key] = self.chart_to_dict(figure)
        
        return converted

    def create_queryable_index(self) -> Dict:
        """
        Create a queryable index of all data.
        
        Returns a dictionary where you can easily look up:
        - All dosage information
        - All survival data
        - All patient characteristics
        - etc.
        """
        index = {
            'dosage_tables': [],
            'patient_characteristics': [],
            'outcomes': [],
            'adverse_events': [],
            'survival_curves': [],
            'other_tables': [],
            'other_figures': []
        }
        
        # Index tables by type
        for table in self.tables:
            table_type = table.get('type', 'other')
            table_dict = self.table_to_dict(table)
            
            if table_type == 'dosage':
                index['dosage_tables'].append(table_dict)
            elif table_type == 'patient_characteristics':
                index['patient_characteristics'].append(table_dict)
            elif table_type == 'outcomes':
                index['outcomes'].append(table_dict)
            elif table_type == 'adverse_events':
                index['adverse_events'].append(table_dict)
            else:
                index['other_tables'].append(table_dict)
        
        # Index figures by type
        for figure in self.figures:
            figure_type = figure.get('type', 'other')
            
            if figure_type == 'survival_curve':
                figure_dict = self.survival_curve_to_dict(figure)
                index['survival_curves'].append(figure_dict)
            else:
                figure_dict = self.chart_to_dict(figure)
                index['other_figures'].append(figure_dict)
        
        return index

    def create_flat_lookup(self) -> Dict[str, Any]:
        """
        Create completely flat key-value lookup.
        
        Example keys:
        - "Baseline_characteristics.Age.Mean" = "65.2"
        - "Survival_analysis.Treatment_A.Median_OS" = "24.1 months"
        """
        flat = {}
        
        # Flatten all tables
        for table in self.tables:
            table_flat = self.table_to_flat_dict(table)
            flat.update(table_flat)
        
        # Flatten survival curves
        for i, figure in enumerate(self.figures):
            if figure.get('type') == 'survival_curve':
                # Use figure number if available, otherwise use title
                figure_number = figure.get('figure_number', '')
                figure_title = figure.get('title', f'Figure_{i+1}')
                
                if figure_number and figure_number != 'Unknown':
                    prefix = figure_number.replace(' ', '_').replace('.', '_')
                else:
                    prefix = figure_title.replace(' ', '_').replace('.', '_')[:50]
                
                # Add p-value
                if figure.get('p_value'):
                    flat[f"{prefix}.p_value"] = figure['p_value']
                
                # Add title for reference
                flat[f"{prefix}.title"] = figure_title
                flat[f"{prefix}.page"] = figure.get('page', 0)
                
                # Add group data
                for group in figure.get('groups', []):
                    group_name = group.get('name', 'Unknown').replace(' ', '_').replace('.', '_')
                    
                    if group.get('median_survival_months'):
                        flat[f"{prefix}.{group_name}.median_survival_months"] = group['median_survival_months']
                    
                    # Add survival rates
                    for timepoint, rate in group.get('survival_rates', {}).items():
                        flat[f"{prefix}.{group_name}.survival_rate_{timepoint}"] = rate
        
        return flat

    def save_dictionaries(self, output_file: str):
        """Save all converted dictionaries to JSON."""
        output = {
            'tables_as_dicts': self.convert_all_tables(),
            'figures_as_dicts': self.convert_all_figures(),
            'queryable_index': self.create_queryable_index(),
            'flat_lookup': self.create_flat_lookup()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved dictionaries to: {output_file}")
        return output


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python table_chart_to_dict.py <extraction_file>")
        print("\nExample:")
        print("  python table_chart_to_dict.py extracted_figures_tables/document_figures_tables.json")
        return
    
    extraction_file = sys.argv[1]
    
    if not Path(extraction_file).exists():
        print(f"✗ File not found: {extraction_file}")
        return
    
    print("\n" + "="*70)
    print("TABLE/CHART TO DICTIONARY CONVERTER")
    print("="*70 + "\n")
    
    # Initialize converter
    converter = TableChartDictConverter(extraction_file)
    
    # Create output filename
    input_path = Path(extraction_file)
    output_file = input_path.parent / f"{input_path.stem}_dicts.json"
    
    # Convert and save
    output = converter.save_dictionaries(str(output_file))
    
    # Print summary
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    
    tables_dict = output['tables_as_dicts']
    figures_dict = output['figures_as_dicts']
    index = output['queryable_index']
    flat = output['flat_lookup']
    
    print(f"\n📊 Tables converted: {len(tables_dict)}")
    for table_key, table_data in tables_dict.items():
        title = table_data['metadata']['title']
        page = table_data['metadata']['page']
        print(f"  - {table_key}")
        print(f"    Title: {title}")
        print(f"    Page: {page}")
    
    print(f"\n📈 Figures converted: {len(figures_dict)}")
    for fig_key, fig_data in figures_dict.items():
        title = fig_data['metadata']['title']
        page = fig_data['metadata']['page']
        print(f"  - {fig_key}")
        print(f"    Title: {title}")
        print(f"    Page: {page}")
    
    print(f"\n🔍 Queryable Index:")
    print(f"  - Dosage tables: {len(index['dosage_tables'])}")
    print(f"  - Patient characteristics: {len(index['patient_characteristics'])}")
    print(f"  - Outcomes: {len(index['outcomes'])}")
    print(f"  - Adverse events: {len(index['adverse_events'])}")
    print(f"  - Survival curves: {len(index['survival_curves'])}")
    
    print(f"\n🗂️  Flat lookup: {len(flat)} key-value pairs")
    
    # Show some example keys
    if flat:
        print("\n  Example keys:")
        for i, key in enumerate(list(flat.keys())[:5]):
            print(f"    - {key} = {flat[key]}")
    
    print(f"\n✓ All data structures saved to: {output_file}")


if __name__ == "__main__":
    main()
