// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'database.dart';

// ignore_for_file: type=lint
class $DataPointsTable extends DataPoints
    with TableInfo<$DataPointsTable, DataPoint> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $DataPointsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
    'id',
    aliasedName,
    false,
    hasAutoIncrement: true,
    type: DriftSqlType.int,
    requiredDuringInsert: false,
    defaultConstraints: GeneratedColumn.constraintIsAlways(
      'PRIMARY KEY AUTOINCREMENT',
    ),
  );
  static const VerificationMeta _latitudeMeta = const VerificationMeta(
    'latitude',
  );
  @override
  late final GeneratedColumn<double> latitude = GeneratedColumn<double>(
    'latitude',
    aliasedName,
    false,
    type: DriftSqlType.double,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _longitudeMeta = const VerificationMeta(
    'longitude',
  );
  @override
  late final GeneratedColumn<double> longitude = GeneratedColumn<double>(
    'longitude',
    aliasedName,
    false,
    type: DriftSqlType.double,
    requiredDuringInsert: true,
  );
  static const VerificationMeta _pointIdMeta = const VerificationMeta(
    'pointId',
  );
  @override
  late final GeneratedColumn<String> pointId = GeneratedColumn<String>(
    'point_id',
    aliasedName,
    false,
    type: DriftSqlType.string,
    requiredDuringInsert: true,
  );
  @override
  List<GeneratedColumn> get $columns => [id, latitude, longitude, pointId];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'data_points';
  @override
  VerificationContext validateIntegrity(
    Insertable<DataPoint> instance, {
    bool isInserting = false,
  }) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('latitude')) {
      context.handle(
        _latitudeMeta,
        latitude.isAcceptableOrUnknown(data['latitude']!, _latitudeMeta),
      );
    } else if (isInserting) {
      context.missing(_latitudeMeta);
    }
    if (data.containsKey('longitude')) {
      context.handle(
        _longitudeMeta,
        longitude.isAcceptableOrUnknown(data['longitude']!, _longitudeMeta),
      );
    } else if (isInserting) {
      context.missing(_longitudeMeta);
    }
    if (data.containsKey('point_id')) {
      context.handle(
        _pointIdMeta,
        pointId.isAcceptableOrUnknown(data['point_id']!, _pointIdMeta),
      );
    } else if (isInserting) {
      context.missing(_pointIdMeta);
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  DataPoint map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return DataPoint(
      id: attachedDatabase.typeMapping.read(
        DriftSqlType.int,
        data['${effectivePrefix}id'],
      )!,
      latitude: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}latitude'],
      )!,
      longitude: attachedDatabase.typeMapping.read(
        DriftSqlType.double,
        data['${effectivePrefix}longitude'],
      )!,
      pointId: attachedDatabase.typeMapping.read(
        DriftSqlType.string,
        data['${effectivePrefix}point_id'],
      )!,
    );
  }

  @override
  $DataPointsTable createAlias(String alias) {
    return $DataPointsTable(attachedDatabase, alias);
  }
}

class DataPoint extends DataClass implements Insertable<DataPoint> {
  final int id;
  final double latitude;
  final double longitude;
  final String pointId;
  const DataPoint({
    required this.id,
    required this.latitude,
    required this.longitude,
    required this.pointId,
  });
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['latitude'] = Variable<double>(latitude);
    map['longitude'] = Variable<double>(longitude);
    map['point_id'] = Variable<String>(pointId);
    return map;
  }

  DataPointsCompanion toCompanion(bool nullToAbsent) {
    return DataPointsCompanion(
      id: Value(id),
      latitude: Value(latitude),
      longitude: Value(longitude),
      pointId: Value(pointId),
    );
  }

  factory DataPoint.fromJson(
    Map<String, dynamic> json, {
    ValueSerializer? serializer,
  }) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return DataPoint(
      id: serializer.fromJson<int>(json['id']),
      latitude: serializer.fromJson<double>(json['latitude']),
      longitude: serializer.fromJson<double>(json['longitude']),
      pointId: serializer.fromJson<String>(json['pointId']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'latitude': serializer.toJson<double>(latitude),
      'longitude': serializer.toJson<double>(longitude),
      'pointId': serializer.toJson<String>(pointId),
    };
  }

  DataPoint copyWith({
    int? id,
    double? latitude,
    double? longitude,
    String? pointId,
  }) => DataPoint(
    id: id ?? this.id,
    latitude: latitude ?? this.latitude,
    longitude: longitude ?? this.longitude,
    pointId: pointId ?? this.pointId,
  );
  DataPoint copyWithCompanion(DataPointsCompanion data) {
    return DataPoint(
      id: data.id.present ? data.id.value : this.id,
      latitude: data.latitude.present ? data.latitude.value : this.latitude,
      longitude: data.longitude.present ? data.longitude.value : this.longitude,
      pointId: data.pointId.present ? data.pointId.value : this.pointId,
    );
  }

  @override
  String toString() {
    return (StringBuffer('DataPoint(')
          ..write('id: $id, ')
          ..write('latitude: $latitude, ')
          ..write('longitude: $longitude, ')
          ..write('pointId: $pointId')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(id, latitude, longitude, pointId);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is DataPoint &&
          other.id == this.id &&
          other.latitude == this.latitude &&
          other.longitude == this.longitude &&
          other.pointId == this.pointId);
}

class DataPointsCompanion extends UpdateCompanion<DataPoint> {
  final Value<int> id;
  final Value<double> latitude;
  final Value<double> longitude;
  final Value<String> pointId;
  const DataPointsCompanion({
    this.id = const Value.absent(),
    this.latitude = const Value.absent(),
    this.longitude = const Value.absent(),
    this.pointId = const Value.absent(),
  });
  DataPointsCompanion.insert({
    this.id = const Value.absent(),
    required double latitude,
    required double longitude,
    required String pointId,
  }) : latitude = Value(latitude),
       longitude = Value(longitude),
       pointId = Value(pointId);
  static Insertable<DataPoint> custom({
    Expression<int>? id,
    Expression<double>? latitude,
    Expression<double>? longitude,
    Expression<String>? pointId,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (latitude != null) 'latitude': latitude,
      if (longitude != null) 'longitude': longitude,
      if (pointId != null) 'point_id': pointId,
    });
  }

  DataPointsCompanion copyWith({
    Value<int>? id,
    Value<double>? latitude,
    Value<double>? longitude,
    Value<String>? pointId,
  }) {
    return DataPointsCompanion(
      id: id ?? this.id,
      latitude: latitude ?? this.latitude,
      longitude: longitude ?? this.longitude,
      pointId: pointId ?? this.pointId,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (latitude.present) {
      map['latitude'] = Variable<double>(latitude.value);
    }
    if (longitude.present) {
      map['longitude'] = Variable<double>(longitude.value);
    }
    if (pointId.present) {
      map['point_id'] = Variable<String>(pointId.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('DataPointsCompanion(')
          ..write('id: $id, ')
          ..write('latitude: $latitude, ')
          ..write('longitude: $longitude, ')
          ..write('pointId: $pointId')
          ..write(')'))
        .toString();
  }
}

abstract class _$AppDatabase extends GeneratedDatabase {
  _$AppDatabase(QueryExecutor e) : super(e);
  $AppDatabaseManager get managers => $AppDatabaseManager(this);
  late final $DataPointsTable dataPoints = $DataPointsTable(this);
  @override
  Iterable<TableInfo<Table, Object?>> get allTables =>
      allSchemaEntities.whereType<TableInfo<Table, Object?>>();
  @override
  List<DatabaseSchemaEntity> get allSchemaEntities => [dataPoints];
}

typedef $$DataPointsTableCreateCompanionBuilder =
    DataPointsCompanion Function({
      Value<int> id,
      required double latitude,
      required double longitude,
      required String pointId,
    });
typedef $$DataPointsTableUpdateCompanionBuilder =
    DataPointsCompanion Function({
      Value<int> id,
      Value<double> latitude,
      Value<double> longitude,
      Value<String> pointId,
    });

class $$DataPointsTableFilterComposer
    extends Composer<_$AppDatabase, $DataPointsTable> {
  $$DataPointsTableFilterComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnFilters<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get latitude => $composableBuilder(
    column: $table.latitude,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<double> get longitude => $composableBuilder(
    column: $table.longitude,
    builder: (column) => ColumnFilters(column),
  );

  ColumnFilters<String> get pointId => $composableBuilder(
    column: $table.pointId,
    builder: (column) => ColumnFilters(column),
  );
}

class $$DataPointsTableOrderingComposer
    extends Composer<_$AppDatabase, $DataPointsTable> {
  $$DataPointsTableOrderingComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  ColumnOrderings<int> get id => $composableBuilder(
    column: $table.id,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get latitude => $composableBuilder(
    column: $table.latitude,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<double> get longitude => $composableBuilder(
    column: $table.longitude,
    builder: (column) => ColumnOrderings(column),
  );

  ColumnOrderings<String> get pointId => $composableBuilder(
    column: $table.pointId,
    builder: (column) => ColumnOrderings(column),
  );
}

class $$DataPointsTableAnnotationComposer
    extends Composer<_$AppDatabase, $DataPointsTable> {
  $$DataPointsTableAnnotationComposer({
    required super.$db,
    required super.$table,
    super.joinBuilder,
    super.$addJoinBuilderToRootComposer,
    super.$removeJoinBuilderFromRootComposer,
  });
  GeneratedColumn<int> get id =>
      $composableBuilder(column: $table.id, builder: (column) => column);

  GeneratedColumn<double> get latitude =>
      $composableBuilder(column: $table.latitude, builder: (column) => column);

  GeneratedColumn<double> get longitude =>
      $composableBuilder(column: $table.longitude, builder: (column) => column);

  GeneratedColumn<String> get pointId =>
      $composableBuilder(column: $table.pointId, builder: (column) => column);
}

class $$DataPointsTableTableManager
    extends
        RootTableManager<
          _$AppDatabase,
          $DataPointsTable,
          DataPoint,
          $$DataPointsTableFilterComposer,
          $$DataPointsTableOrderingComposer,
          $$DataPointsTableAnnotationComposer,
          $$DataPointsTableCreateCompanionBuilder,
          $$DataPointsTableUpdateCompanionBuilder,
          (
            DataPoint,
            BaseReferences<_$AppDatabase, $DataPointsTable, DataPoint>,
          ),
          DataPoint,
          PrefetchHooks Function()
        > {
  $$DataPointsTableTableManager(_$AppDatabase db, $DataPointsTable table)
    : super(
        TableManagerState(
          db: db,
          table: table,
          createFilteringComposer: () =>
              $$DataPointsTableFilterComposer($db: db, $table: table),
          createOrderingComposer: () =>
              $$DataPointsTableOrderingComposer($db: db, $table: table),
          createComputedFieldComposer: () =>
              $$DataPointsTableAnnotationComposer($db: db, $table: table),
          updateCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                Value<double> latitude = const Value.absent(),
                Value<double> longitude = const Value.absent(),
                Value<String> pointId = const Value.absent(),
              }) => DataPointsCompanion(
                id: id,
                latitude: latitude,
                longitude: longitude,
                pointId: pointId,
              ),
          createCompanionCallback:
              ({
                Value<int> id = const Value.absent(),
                required double latitude,
                required double longitude,
                required String pointId,
              }) => DataPointsCompanion.insert(
                id: id,
                latitude: latitude,
                longitude: longitude,
                pointId: pointId,
              ),
          withReferenceMapper: (p0) => p0
              .map((e) => (e.readTable(table), BaseReferences(db, table, e)))
              .toList(),
          prefetchHooksCallback: null,
        ),
      );
}

typedef $$DataPointsTableProcessedTableManager =
    ProcessedTableManager<
      _$AppDatabase,
      $DataPointsTable,
      DataPoint,
      $$DataPointsTableFilterComposer,
      $$DataPointsTableOrderingComposer,
      $$DataPointsTableAnnotationComposer,
      $$DataPointsTableCreateCompanionBuilder,
      $$DataPointsTableUpdateCompanionBuilder,
      (DataPoint, BaseReferences<_$AppDatabase, $DataPointsTable, DataPoint>),
      DataPoint,
      PrefetchHooks Function()
    >;

class $AppDatabaseManager {
  final _$AppDatabase _db;
  $AppDatabaseManager(this._db);
  $$DataPointsTableTableManager get dataPoints =>
      $$DataPointsTableTableManager(_db, _db.dataPoints);
}
