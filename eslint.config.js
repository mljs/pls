import { defineConfig, globalIgnores } from 'eslint/config';
import cheminfo from 'eslint-config-cheminfo';

export default defineConfig([globalIgnores(['coverage', 'lib']), cheminfo]);
