import os
import ast
import inspect
import importlib
from collections.abc import Iterable
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import BaseEnsemble
import manalake

class PipelineBuilder:
    """
    Base class for building machine learning pipelines.
    """
    def __init__(self, column_names=None, base_dir=None):
        if self.__class__ is PipelineBuilder:
            raise TypeError("Cannot instantiate abstract class PipelineBuilder.")
        
        self.column_names = column_names if column_names is not None else []
        self.base_dir = base_dir if base_dir is not None else os.getcwd()
        self._blueprints = {}
               
    def _inspect_pipelines(self):
        """
        Display all pipelines stored in the _blueprints.
        """
        if not self._blueprints:
            print("No pipelines have been created yet.")
        else:
            print("Current Pipelines:")
            for name, steps in self._blueprints.items():
                print(f"Pipeline '{name}':")
                for idx, step in enumerate(steps['sections'], 1):
                    print(f"  {idx}. {step}")
    
        eligible_params = self._get_class_params(section_class)
        return True, section_class, eligible_params

    def _get_class_params(self, section_class):
        """
        Retrieves eligible parameters for a class using inspection.
        """
        params = inspect.signature(section_class).parameters
        return list(params.keys())
    
    def _retrieve_class_from_scope(self, section_name):
        """
        Retrieves the class object from global or local scope.
        """
        section_class = globals().get(section_name) or locals().get(section_name)
        if section_class is None:
            print(f"Error: '{section_name}' is not defined in the current scope. Try again.")
            return None
        if not callable(section_class):
            print(f"Error: '{section_name}' is not callable. Try again.")
            return None
        return section_class
        
    def _validate_section(self, section_name, is_predictor):
        """
        Validates if a given section can be added to the pipeline.
        """
        section_class = self._retrieve_class_from_scope(section_name)
        if not section_class:
            return False, None, []
    
        if is_predictor:
            valid, section_class  = self._validate_predictor(section_class)
        else:
            valid, section_class = self._validate_transformer(section_class)
 
        if not valid:
            return False, None, []
        else:
            valid_params = self._get_class_params(section_class)
            return valid, section_class, valid_params
            
    def _validate_predictor(self, section_class):
        """
        Validates if the given class is a predictor.
        """
        if not hasattr(section_class, 'predict') or \
           not any(base.__name__ in ['BaseEstimator', 'BaseEnsemble'] for base in inspect.getmro(section_class)):
            print(f"Error: '{section_class.__name__}' is not a valid predictor "
                  f"(no 'predict' method or not derived from 'BaseEstimator' or 'BaseEnsemble'). Try again.")
            return False, None
        return True, section_class

    def _validate_transformer(self, section_class):
        """
        Validates if the given class is a transformer.
        """
        if not hasattr(section_class, 'transform') or \
           'TransformerMixin' not in [base.__name__ for base in inspect.getmro(section_class)]:
            print(f"Error: '{section_class.__name__}' is not a valid transformer "
                  f"(no 'transform' method or not derived from 'TransformerMixin'). Try again.")
            return False, None
        return True, section_class

class IterPlumber(PipelineBuilder):
    """
    Interactive pipeline builder allowing step-by-step creation and management of multiple pipelines.
    """
    def __init__(self, column_names=None, base_dir=None):
        # Initialize parent class with the provided arguments
        super().__init__(column_names=column_names, base_dir=base_dir)

        # Initialize available sections as an empty 
        self.available_sections = {
            'transformers': {}, 
            'predictors' : {}
        } # Tracks user-created components not yet in use
        
    def run_pipes(self):
        """
        Main method to manage the creation of pipeline instructions for multiple pipelines iteratively.
        """
        while True:
            user_input = input(
                """
    Welcome to IterPlumber! Please select an option:
    1. Create a new pipeline
    2. View current pipelines
    3. Finalize blueprints
    4. Cancel and exit
    Your choice: """.strip()).strip()
    
            if user_input == '1':
                # Initialize the current pipeline as an empty dictionary
                print("\nFollow the instructions to iteratively build a pipeline.")
                current_pipeline = self._build_pipeline()
    
                # If a pipeline was successfully built
                if current_pipeline:
                    pipeline_name = current_pipeline['name']
                    self._blueprints[pipeline_name] = current_pipeline
                    print(f"Pipeline '{pipeline_name}' saved successfully.")
                else:
                    print("Pipeline creation was cancelled or not completed.")
    
            elif user_input == '2':
                # View current pipelines
                if self._blueprints:
                    print("\nCurrent pipelines:")
                    for name, pipeline in self._blueprints.items():
                        print(f"- {name}: {pipeline}")
                else:
                    print("\nNo pipelines have been created yet.")
    
            elif user_input == '3':
                # Finalize and save the pipelines
                print("\nFinalizing and saving pipelines...")
                self._finalize_blueprints()
                print("Pipelines finalized successfully. Goodbye!")
                break
    
            elif user_input == '4':
                # Cancel and exit
                print("\nCancelling all operations. Goodbye!")
                break
    
            else:
                # Handle invalid input
                print("\nInvalid choice. Please select a valid option (1-4).")


    def _build_pipeline(self):
        """
        Helper method to interactively build pipeline instructions for a single pipeline.
        """
        self.current_column_names = {col: col for col in self.column_names}  # Keeps column mapping updated dynamically
        
        # Empty pipeline is instanced
        section_id = 0
        section_number = 0
        
        this_pipeline = {
            'name': ''
            , 'n_sections': 0
            , 'sections': {}
        }

        # Flow controls for pipeline assembly
        has_transformer = False
        has_predictor = False
    
        while True:
            print("\n--- Pipeline Creation Menu ---")
            print("To build a pipeline, add at least one transformer and one predictor section."
                  "The last section must always be a predictor. Sections must be built before they can be added.")
            print("\nOptions:")
            print("1. Build a transformer section")
            print("   - A transformer applies preprocessing steps to your data, such as scaling, encoding, or imputation.")
            print("2. Build a predictor")
            print("   - A predictor is the final step in the pipeline, such as a regression model or classifier.")
            print("3. Add a section to the pipeline")
            print("   - Use this to integrate a previously defined transformer or predictor into the pipeline.")
            print("4. View current pipeline")
            print("   - Displays the steps currently added to this pipeline, in the order they will be applied.")
            print("5. View current available sections")
            print("   - Displays the steps currently added to this pipeline, in the order they will be applied.")
            print("6. Finish and save this pipeline")
            print("   - Completes the pipeline creation process and saves the current pipeline.")
            print("7. Cancel this pipeline")
            print("   - Discards the current pipeline and returns to the main menu.")
            
            choice = input("Enter your choice: ").strip()
            
            if choice == '1':
                print("Building transformer section.")
                self._handle_section()
        
            elif choice == '2':
                print("Building predictor.")
                self._handle_section(is_predictor=True)
                
            elif choice == '3':
                if has_predictor:
                    print('Unable to add more sections, pipeline already has a predictor')
                    break
                
                if self.available_sections['transformers'] or self.available_sections['predictors']:
                    while True:
                        print("Adding section to the pipeline.")
                        print("Select which type of section you wish to add to the pipeline")
                        print("1. Add a transformer")
                        print("2. Add a predictor")
                        print("3. Add a column_transformer (requires a transformer)")
                        print("4. Go back to Pipeline creation menu.")
    
                        self._view_sections()
                        
                        choice = input("Enter your choice: ").strip()
                        if choice == '1':
                            if self.available_sections['transformers']:
                                has_transformer = self._add_section(this_pipeline, is_predictor=False)
                            else:
                                print('No transfomers available to add.')
                            
                        elif choice == '2':
                            if self.available_sections['predictors']:
                                has_predictor = self._add_section(this_pipeline, is_predictor=True)
                            else:
                                print('No predictors available to add.')
                        elif choice == '3':
                            if self.available_sections['transformers']:
                                has_transformer = self._add_section(this_pipeline, is_column_transformer=True)

                        elif choice == '4':
                            print('Returning to previous menu.')
                            break
                        else:
                            print('No transfomers available to add.')
                            
                else:
                    print("No pipes available to add.")
                
            elif choice == '4':
                self._view_pipeline(this_pipeline)
                
            elif choice == '5':
                self._view_sections()
                
            elif choice == '6':
                if has_predictor:
                    predictor_key = max(this_pipeline['sections'].keys())
                    this_pipeline['name'] = this_pipeline['sections'][predictor_key]['name']
                
                    print("Saving the current pipeline with predictor:", this_pipeline['name'] )
                    return this_pipeline
                    
                else:
                    print("Cannot store empty or incomplete pipeline blueprints. Please try again.")
                    
            elif choice == '7': 
                print("Cancelling the current pipeline and returning to the main menu.")
                input("Press any key to continue")
                return None
                
            else:
                print("Invalid choice. Please try again.")
            
            input("Press any key to continue")
        
                
    def _handle_section(self, is_predictor=False):
        """Handles flow into _build_section which itself provides an iterative procedure to generate pipelines, receives a section in return and parses it into the available_sections"""
        if is_predictor:
            string = "predictors"
        else:
            string = "transformers"
        
        section_name = input(f"Enter the {string} class name: ").strip()   

        if section_name.lower() in self.available_sections[string]:
            print(f'Failed to create {section_name}. Another section with that name already exists')
            valid = False
        else:
            valid, section_class, eligible_params = self._validate_section(section_name, is_predictor)
        
        if valid:
            section = self._build_section(section_name, section_class, eligible_params)

            self.available_sections[string][section['_name']] = {
                'name': section['_name'],
                'class': section['_class'],
                'args': section['_args'],
                'grid': section['_grid'],
                'columns' : section['_columns']
            }
            
        else:
            input(f"Failed to build {string} section. Press any key to return to the previous menu")
                
    def _build_section(self, section_name, section_class, eligible_params):
        """
        Prompts the user to provide values for eligible parameters of a section
        and validates them by instantiating the class and calling its `fit` method.
        """
        # Try to print documentation to assist the user
        try:
            print(section_class.__doc__)
        except Exception as e:
            print(f"\nUnexpected issue with docstring: {e}")
    
        print(f"Now building '{section_name.lower()}'. Ensure that parameter values are correct (Docstring above).")
    
        params = {}
        grid_params = {}
               

        for param in eligible_params:
            user_input = input(f"Enter a value for '{param}' (or an iterable like a list or range for grid search if applicable): ")
            try:
                if user_input == '':
                    user_input = None
                else:
                    parsed_input = eval(user_input)
            except (ValueError, SyntaxError):
                parsed_input = user_input
        
            if not user_input:
                continue
        
            if isinstance(parsed_input, (dict, list)) or isinstance(parsed_input, Iterable):
                grid_params[param] = parsed_input
            else:
                params[param] = parsed_input
        
        if not grid_params:
            max_iterations = 1
            grid_iterators = {}
        else:
            grid_iterators = {param: iter(values) for param, values in grid_params.items()}
             
        # Testing loop
        number_iterators = len(grid_params.keys())
        number_stop_iterators = 0
        current_args = params.copy()
        try:
            while number_stop_iterators != number_iterators:
                number_stop_iterators = 0
                for param, iterator in grid_iterators.items():
                    try:
                        current_args[param] = next(iterator)
                    except StopIteration:
                        number_stop_iterators += 1
                        
                print(f"Testing with arguments: {current_args}")
                dummy_data_x, dummy_data_y = [[0]], [[0]]  # Placeholder data
                section_instance = section_class(**current_args)
                try:
                    section_instance.fit(dummy_data_x)  # Validate the instance
                except Exception as e:
                    section_instance.fit(dummy_data_x, dummy_data_y)
        except Exception as e:
            print(f"Failed with arguments {current_args}: {e}")
            retry = input("Invalid arguments detected, try again? (y/n): ").strip().lower()
            if retry != 'y':
                print("Exiting pipeline creation.")
                return None
            else:
                # Retry the entire section
                return self._build_section(section_name, section_class, eligible_params)
    
        print(f"Successfully tested all arguments,")
        section = {
            '_name': section_class.__name__.lower(),
            '_class': section_class.__name__,
            '_args': params,
            '_grid': grid_params,
            '_columns': []
        }
    
        return section

    
    def _add_section(self, pipeline, is_predictor=False, is_column_transformer=False):
        """
        Add a section to the pipeline, either as a predictor or a transformer.
        Handles column selection for column transformers iteratively.
    
        Args:
            pipeline (dict): The current pipeline being built.
            is_predictor (bool): Whether the section is a predictor. Defaults to False.
            is_column_transformer (bool): Whether the section is a column transformer. Defaults to False.
    
        Returns:
            bool: True if at least one section was added, False otherwise.
        """
        section_type = 'predictors' if is_predictor else 'transformers'
        selectable_sections = self.available_sections[section_type].copy()
        selected_sections = {}
        
        # For column transformers, maintain a list of selectable columns
        selectable_columns = self.column_names.copy() if is_column_transformer else None
        has_more_to_add = True
        at_least_one_selected = False
        section_number = 0  # Internal section number (increments for column transformers)
    
        while has_more_to_add:
            # Step 1: Handle column selection if necessary
            selected_columns = None
            if is_column_transformer:
                selected_columns = self._select_columns(selectable_columns)
                if not selected_columns:
                    print("No columns selected. Returning to the previous menu.")
                    return False
                # Update the remaining selectable columns
                selectable_columns = [col for col in selectable_columns if col not in selected_columns]
    
            # Step 2: Handle section selection
            section_name, section_details = self._select_section(selectable_sections)
            if not section_name:
                print("No section selected. Returning to the previous menu.")
                return False
                        
            # Add columns if this is a column transformer
            if is_column_transformer:
                section_details['name'] = section_name
                section_details['columns'] = selected_columns
                print(f"Assigned columns {selected_columns} to {section_name}.")
                
            
            # Increment for each section selected
            section_number += 1  
            
            # Store the selected section
            selected_sections[section_number] = section_details
            selectable_sections.pop(section_name, None)  # Remove from available sections
            
            at_least_one_selected = True
    
            # Step 3: Decide whether to add more transformers to this column transformer
            if is_column_transformer and selectable_columns and selectable_sections:
                add_more = input("Add another transformer to the column transformer? (y/n): ").strip().lower()
                if add_more != 'y':
                    print("Finalizing current column transformer.")
                    has_more_to_add = False
            else:
                has_more_to_add = False  # No more sections or columns to add
    
        if at_least_one_selected:
            # Update the id before incrementing
            section_id = pipeline['n_sections'] + 1
                
            # Step 4: Add column transformer or section to the pipeline
            if is_column_transformer:
                # Create a composite name for the column transformer
                transformer_name = '_'.join(
                    ['column_transformer'] + [
                        selected_sections[section_number]['name'] for section_number in selected_sections
                    ]
                )
                column_transformer = {
                    'name': transformer_name,
                    'transformers': selected_sections
                }
                # Add the column transformer to the pipeline
                pipeline['sections'][section_id] = column_transformer

            else:
                # Add single predictor/transformer to the pipeline
                section_number = next(iter(selected_sections))  # Extract the single selected section
                pipeline['sections'][section_id] = selected_sections[section_number]
            
            # Remove selected sections from available_sections
            for section_number in selected_sections:
                del self.available_sections[section_type][selected_sections[section_number]['name']]

            # Increment the number of sections
            pipeline['n_sections'] += 1
        
            return True
            
        else:
            print("No sections were successfully added to the pipeline.")
            return False

    def _select_section(self, available_sections):
        while True:
            print(f"Available sections: \n{available_sections.keys()}")
            chosen_section = input("Select a transformer by entering its name (see above)")
            
            if chosen_section not in available_sections.keys():
                print("Failed to match your input with an available section")
                retry = input("Try again y/n?")
                if retry == 'y':
                    continue
                else:
                    print('Returning to the previous menu')
                    return None, None
            else:
                return chosen_section, available_sections[chosen_section]
                    

    def _select_columns(self, remaining_columns):
        """
        Prompt the user to select specific columns for the given transformer section.
        Allows the user to cancel and return to the previous menu.
        """
        while True:
            # Display available columns
            print("\nAvailable columns:")
            for idx, col in enumerate(remaining_columns, start=1):
                print(f"{idx}. {col}")
            
            print("\nEnter column names as a comma-separated list for this transformer.")
            print("Type 'cancel' to return to the previous menu.")
            
            # Get user input
            user_input = input("Enter column names or 'cancel': ").strip()
            
            # Handle cancel option
            if user_input.lower() == 'cancel':
                print("Returning to the previous menu.")
                return None
            
            # Parse the input into a list
            selected_columns = [col.strip() for col in user_input.split(",")]
            
            # Validate selected columns
            invalid_columns = [col for col in selected_columns if col not in remaining_columns]
            if invalid_columns:
                print(f"Invalid columns: {invalid_columns}. Please try again.")
            elif not selected_columns:
                print("No columns entered. Please select at least one column.")
            else:
                return selected_columns
        
    def _view_sections(self):
        for section_type in self.available_sections.items():
            if not section_type:
                print(f"No sections available for type {section_type}")
            else:
                for section in section_type:
                    print(section),
                print()  
                
    def _view_pipeline(self, this_pipeline):
        if not this_pipeline:
            print(" No sections in use/available.")
        else:
            print("Current pipeline:")
            for step_name, step_details in this_pipeline['sections'].items():
                print(f"- {step_name}: {step_details}")

    def _finalize_blueprints(self):
        """
        Finalize all blueprints by saving them using an external auditor.
        """
        if not self._blueprints:
            print("No pipelines were created. Nothing to save.")
            return
    
        print("\nFinalizing and saving all created pipelines...")
        # Placeholder for MLAuditer usage
        self.auditer = manalake.MLAuditer(self.base_dir)
        self.auditer.sign_and_save_config(self._blueprints)
        print("Pipelines saved successfully.")


class AutoPlumber(PipelineBuilder):
    def __init__(self, column_names=None, base_dir=None, *, blueprints):
        # Ensure blueprints are provided
        if blueprints is None:
            raise ValueError("AutoPlumber requires a blueprint to construct the pipeline. "
                             "For iterative pipeline building, use IterPlumber instead.")

        # Call the superclass initializer
        super().__init__(column_names=column_names, base_dir=base_dir)

        # Additional initialization for AutoPlumber
        self._blueprints = blueprints
            
        def run_pipes(self, _blueprints):
            """
            Automatically builds the pipeline based on user instructions.
            """
            print("AutoPlumber: Constructing the pipeline automatically...")
            n_steps = len(_blueprints)
        
            current_step = 0
            for step_name, details in _blueprints.items():
                is_predictor = (current_step == n_steps - 1)
                valid, section_class, eligible_params = self._validate_section(step_name, is_predictor)
                if not valid:
                    raise ValueError(f"Invalid pipeline step at '{step_name}'.")
        
                print(f"Step '{step_name}' validated.")
                self.add_section(step_name, section_class, eligible_params)
                current_step += 1
    
            # Instance MLAuditer
            self.auditer = MLAuditer(self.base_dir)
            # MLAuditer call to sign and save configuration
            self.auditer.sign_and_save_config(self._blueprints)

