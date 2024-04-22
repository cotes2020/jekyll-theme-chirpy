---
published: true
date: 2024-04-22
title: Auto migration command in Laravel
---
```
<?php

namespace App\Console\Commands;

use Doctrine\DBAL\Exception;
use Illuminate\Console\Command;
use Doctrine\DBAL\Schema\Comparator;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\Config;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Str;
use Symfony\Component\Finder\Finder;
use Doctrine\DBAL\DriverManager;
use Doctrine\DBAL\Connection;

class MigrateAutoCommand extends Command
{
    protected $signature = 'migrate:auto {--f|--fresh} {--s|--seed} {--force} {--pretend}';

    protected Connection $conn;

    public function __construct()
    {
        parent::__construct();

        $configPath = 'database.connections.' . strval(Config::get('database.default'));
        $databaseDriverMapping = [
            'mysql' => 'pdo_mysql',
            'mariadb' => 'pdo_mysql',
            'sqlite' => 'pdo_sqlite',
            'pgsql' => 'pdo_pgsql',
            'sqlsrv' => 'pdo_sqlsrv',
        ];
        $connectionParams = [
            'dbname' => Config::get($configPath . '.database'),
            'user' => Config::get($configPath . '.username'),
            'password' => Config::get($configPath . '.password'),
            'host' => Config::get($configPath . '.host'),
            'driver' => $databaseDriverMapping[strval(Config::get('database.default'))],
        ];

        $this->conn = DriverManager::getConnection($connectionParams);
    }

    /**
     * @throws Exception
     */
    final public function handle(): void
    {
        $this->handleTraditionalMigrations();

        $this->handleAutomaticMigrations();

        if ($this->option('seed')) $this->seed();

        $this->info('Automatic migration completed successfully.');
    }

    private function handleTraditionalMigrations(): void
    {
        $command = 'migrate';

        if ($this->option('fresh')) $command .= ':fresh';
        if ($this->option('force')) $command .= ' --force';

        Artisan::call($command, [], $this->getOutput());
    }

    /**
     * @throws Exception
     */
    private function handleAutomaticMigrations(): void
    {
        $models = collect();

        $modelPaths = [
            'App\\Models' => app_path('Models'),
        ];
        foreach ($modelPaths as $namespace => $path) {
            if (!is_dir($path)) continue;

            foreach ((new Finder)->in($path) as $model) {
                $model = $namespace . str_replace(
                        ['/', '.php'],
                        ['\\', ''],
                        Str::after($model->getRealPath(), realpath($path))
                    );

                if (method_exists($model, 'migration'))
                    $models->push([
                        'object' => $object = app($model),
                        'order' => $object->migrationOrder ?? 0,
                    ]);
            }
        }

        foreach ($models->sortBy('order') as $model)
            $this->migrate($model['object']);
    }

    private function logSql(Model $model, array $queries): void
    {
        $this->line("<info>" . get_class($model) . ":</info> {$queries[0]}");
    }

    /**
     * @throws Exception
     */
    private function migrate(Model $model): void
    {
        $modelTable = $model->getTable();
        $pretend = $this->option('pretend'); // option which will display the SQL that will run
        $createTableQuery = static fn(Blueprint $table) => $model->migration($table);

        // alter existing table
        if (Schema::hasTable($modelTable)) {
            $tempTable = time() . "_doctrine_temp_table_$modelTable";

            Schema::dropIfExists($tempTable);
            Schema::create($tempTable, $createTableQuery);

            $schemaManager = $this->conn->createSchemaManager();
            $platform = $this->conn->getDatabasePlatform();
            $comparator = new Comparator($platform);

            $modelTableDetails = $schemaManager->introspectTable($modelTable);
            $tempTableDetails = $schemaManager->introspectTable($tempTable);

            $tableDiff = $comparator->compareTables($modelTableDetails, $tempTableDetails);

            Schema::drop($tempTable);

            if (!$tableDiff->isEmpty()) {
                $queries = $platform->getAlterTableSQL($tableDiff);

                if ($pretend) {
                    $this->logSql($model, $queries);
                } else {
                    $schemaManager->alterTable($tableDiff);
                    $this->line('<info>Table updated:</info> ' . $modelTable);
                }
            }
        } // create new table
        else {
            $createTableCommand = fn() => Schema::create($modelTable, $createTableQuery);
            $queries = Schema::getConnection()->pretend($createTableCommand);

            if ($pretend) {
                $this->logSql($model, $queries);
            } else {
                $createTableCommand();
                $this->line('<info>Table created:</info> ' . $modelTable);
            }
        }
    }

    private function seed(): void
    {
        $command = 'db:seed';

        if ($this->option('force')) $command .= ' --force';

        Artisan::call($command, [], $this->getOutput());
    }
}
```